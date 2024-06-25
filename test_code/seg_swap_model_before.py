import matplotlib

matplotlib.use('Agg')

import yaml
# from argparse import ArgumentParser
import tqdm
import sys

import imageio
import numpy as np
from skimage.transform import resize
from skimage import img_as_ubyte

import torch
from seg_swap.sync_batchnorm import DataParallelWithCallback
import torch.nn.functional as F

from seg_swap.modules.segmentation_module import SegmentationModule
from seg_swap.modules.reconstruction_module import ReconstructionModule
from seg_swap.logger import load_reconstruction_module, load_segmentation_module

from seg_swap.modules.util import AntiAliasInterpolation2d
from seg_swap.modules.dense_motion import DenseMotionNetwork


import cv2

from skimage import img_as_ubyte


if sys.version_info[0] < 3:
    raise Exception("You must use Python 3 or higher. Recommended version is Python 3.7")


class PartSwapGenerator(ReconstructionModule):
    def __init__(self, blend_scale=1, first_order_motion_model=False, **kwargs):
        super(PartSwapGenerator, self).__init__(**kwargs)
        if blend_scale == 1:
            self.blend_downsample = lambda x: x
        else:
            self.blend_downsample = AntiAliasInterpolation2d(1, blend_scale)

        if first_order_motion_model:
            self.dense_motion_network = DenseMotionNetwork()
        else:
            self.dense_motion_network = None

    def forward(self, source_image, target_image, seg_target, seg_source, blend_mask, use_source_segmentation=False):
        # Encoding of source image
        enc_source = self.first(source_image)
        for i in range(len(self.down_blocks)):
            enc_source = self.down_blocks[i](enc_source)

        # Encoding of target image
        enc_target = self.first(target_image)
        for i in range(len(self.down_blocks)):
            enc_target = self.down_blocks[i](enc_target)

        output_dict = {}
        # Compute flow field for source image
        if self.dense_motion_network is None:
            segment_motions = self.segment_motion(seg_target, seg_source)
            segment_motions = segment_motions.permute(0, 1, 4, 2, 3)
            mask = seg_target['segmentation'].unsqueeze(2)
            deformation = (segment_motions * mask).sum(dim=1)
            deformation = deformation.permute(0, 2, 3, 1)
        else:
            motion = self.dense_motion_network(source_image=source_image, seg_target=seg_target,
                                               seg_source=seg_source)
            deformation = motion['deformation']

        # Deform source encoding according to the motion
        enc_source = self.deform_input(enc_source, deformation)

        if self.estimate_visibility:
            if self.dense_motion_network is None:
                visibility = seg_source['segmentation'][:, 1:].sum(dim=1, keepdim=True) * \
                             (1 - seg_target['segmentation'][:, 1:].sum(dim=1, keepdim=True).detach())
                visibility = 1 - visibility
            else:
                visibility = motion['visibility']

            if enc_source.shape[2] != visibility.shape[2] or enc_source.shape[3] != visibility.shape[3]:
                visibility = F.interpolate(visibility, size=enc_source.shape[2:], mode='bilinear')
            enc_source = enc_source * visibility

        blend_mask = self.blend_downsample(blend_mask)
        # If source segmentation is provided use it should be deformed before blending
        if use_source_segmentation:
            blend_mask = self.deform_input(blend_mask, deformation)

        out = enc_target * (1 - blend_mask) + enc_source * blend_mask

        out = self.bottleneck(out)

        for i in range(len(self.up_blocks)):
            out = self.up_blocks[i](out)

        out = self.final(out)
        out = F.sigmoid(out)

        output_dict["prediction"] = out

        return output_dict


def load_checkpoints(config, checkpoint, blend_scale=0.125, first_order_motion_model=False, cpu=False):
    with open(config) as f:
        config = yaml.load(f)

    reconstruction_module = PartSwapGenerator(blend_scale=blend_scale,
                                              first_order_motion_model=first_order_motion_model,
                                              **config['model_params']['reconstruction_module_params'],
                                              **config['model_params']['common_params'])

    if not cpu:
        reconstruction_module.cuda()

    segmentation_module = SegmentationModule(**config['model_params']['segmentation_module_params'],
                                             **config['model_params']['common_params'])
    if not cpu:
        segmentation_module.cuda()

    if cpu:
        checkpoint = torch.load(checkpoint, map_location=torch.device('cpu'))
    else:
        checkpoint = torch.load(checkpoint)

    load_reconstruction_module(reconstruction_module, checkpoint)
    load_segmentation_module(segmentation_module, checkpoint)

    if not cpu:
        reconstruction_module = DataParallelWithCallback(reconstruction_module)
        segmentation_module = DataParallelWithCallback(segmentation_module)

    reconstruction_module.eval()
    segmentation_module.eval()

    return reconstruction_module, segmentation_module


def load_face_parser(cpu=False):
    # from face_parsing.model import BiSeNet
    from parsing_model.model import BiSeNet


    face_parser = BiSeNet(n_classes=19)
    if not cpu:
       face_parser.cuda()
       face_parser.load_state_dict(torch.load('parsing_model/checkpoint/79999_iter.pth'))
    else:
       face_parser.load_state_dict(torch.load('parsing_model/checkpoint/79999_iter.pth', map_location=torch.device('cpu')))
 
    face_parser.eval()

    mean = torch.Tensor(np.array([0.485, 0.456, 0.406], dtype=np.float32)).view(1, 3, 1, 1)
    std = torch.Tensor(np.array([0.229, 0.224, 0.225], dtype=np.float32)).view(1, 3, 1, 1)

    if not cpu:
        face_parser.mean = mean.cuda()
        face_parser.std = std.cuda()
    else:
        face_parser.mean = mean
        face_parser.std = std
 
    return face_parser


def make_video(swap_index, source_image, target_video, reconstruction_module, segmentation_module, face_parser=None,
               hard=False, use_source_segmentation=False, cpu=False):
    assert type(swap_index) == list
    with torch.no_grad():


        source = torch.tensor(source_image[np.newaxis].astype(np.float32)).permute(0, 3, 1, 2)
        if not cpu:
            source = source.cuda()
            print("Oke")
        seg_source = segmentation_module(source)
        # source = torch.tensor(source_image[np.newaxis].astype(np.float32)).permute(0, 3, 1, 2)
        # if not cpu:
        #     source = source.cuda()
        # seg_source = segmentation_module(source)

        # target = torch.tensor(np.array(target_video)[np.newaxis].astype(np.float32)).permute(0, 4, 1, 2, 3)
        # for frame_idx in tqdm(range(target.shape[2])):
        # target_frame = target[:, :, frame_idx]
        target_frame = torch.tensor(target_video[np.newaxis].astype(np.float32)).permute(0, 3, 1, 2)

        if not cpu:
            target_frame = target_frame.cuda()
        seg_target = segmentation_module(target_frame)

        # Computing blend mask
        if face_parser is not None:
            blend_mask = F.interpolate(source if use_source_segmentation else target_frame, size=(512, 512))
            blend_mask = (blend_mask - face_parser.mean) / face_parser.std
            blend_mask = torch.softmax(face_parser(blend_mask)[0], dim=1)
        else:
            blend_mask = seg_source['segmentation'] if use_source_segmentation else seg_target['segmentation']

        blend_mask = blend_mask[:, swap_index].sum(dim=1, keepdim=True)
        if hard:
            blend_mask = (blend_mask > 0.5).type(blend_mask.type())

        out = reconstruction_module(source, target_frame, seg_source=seg_source, seg_target=seg_target,
                                    blend_mask=blend_mask, use_source_segmentation=use_source_segmentation)

        # predictions.append(np.transpose(out['prediction'].data.cpu().numpy(), [0, 2, 3, 1])[0])
        predictions = np.transpose(out['prediction'].data.cpu().numpy(), [0, 2, 3, 1])[0]

    return predictions




def seg_swap_model(opt, box_source, box_target):
    # parser = ArgumentParser()
    # parser.add_argument("--config", required=True, help="path to config")
    # parser.add_argument("--checkpoint", default='vox-cpk.pth.tar', help="path to checkpoint to restore")

    # parser.add_argument("--source_image", default='sup-mat/source.png', help="path to source image")
    # parser.add_argument("--target_video", default='sup-mat/source.png', help="path to target video")
    # # parser.add_argument("--result_video", default='result.mp4', help="path to output")

    # parser.add_argument("--swap_index", default="1,2,5", type=lambda x: list(map(int, x.split(','))),
    #                     help='index of swaped parts')
    # parser.add_argument("--hard", action="store_true", help="use hard segmentation labels for blending")
    # parser.add_argument("--use_source_segmentation", action="store_true", help="use source segmentation for swaping")
    # parser.add_argument("--first_order_motion_model", action="store_true", help="use first order model for alignment")
    # parser.add_argument("--supervised", action="store_true",
    #                     help="use supervised segmentation labels for blending. Only for faces.")

    # parser.add_argument("--cpu", action="store_true", help="cpu mode")
    # parser.add_argument("--num_seg", default=3, help="number of segment times")



    # cpu = False

    source_image = box_source 

    print ('hey boss')
    print (source_image.shape)
    print (type(source_image))
    cv2.imwrite ('output/source.jpg', source_image)

    target_video = box_target 
    print (target_video.shape)
    print (type(target_video))
    cv2.imwrite ('output/target.jpg', target_video)



    source_image = resize(source_image, (256, 256))[..., :3]

    # target_video = imageio.imread(opt.target_video)

    # target_video = resize(target_video, (256, 256))[..., :3]

    blend_scale = (256 / 4) / 512 if opt.supervised else 1
    reconstruction_module, segmentation_module = load_checkpoints(opt.config, opt.checkpoint, blend_scale=blend_scale, 
                                                                  first_order_motion_model=opt.first_order_motion_model, cpu=opt.cpu)

    if opt.supervised:
        face_parser = load_face_parser(opt.cpu)
    else:
        face_parser = None
    # predictions = make_video(opt.swap_index, source_image, target_video, reconstruction_module, segmentation_module,
    #                          face_parser, hard=opt.hard, use_source_segmentation=opt.use_source_segmentation, cpu=opt.cpu)

    # with torch.no_grad():
    #     source = torch.tensor(source_image[np.newaxis].astype(np.float32)).permute(0, 3, 1, 2)
    #     if not cpu:
    #         source = source.cuda()
    #         print("Oke")
    #     seg_source = segmentation_module(source)

    # num_cameras = cv2.VideoCapture(0).get(cv2.CAP_PROP_MODE)
    
    # current_camera_index = 0
    
    # cap = cv2.VideoCapture(current_camera_index)


    # while True:
    #     # Read the current frame from the webcam

    #     ret, frame = cap.read()
        
    #     # frame swap 
    #     # imageio.read returns a numpy array BGR and a meta attribute 
    #     # try to simulate what imageio can do 
    # target_video = cv2.imread (opt.target_video)

    # target_video = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) 


    target_video = resize(target_video, (256, 256))[..., :3]
    
    # predictions = target_video
    print (target_video.shape)
    print (type(target_video))

    # predictions = cv2.cvtColor(target_video, cv2.COLOR_BGR2RGB) 

    predictions = target_video

    

    # for i in range (opt.num_seg):
    #     predictions = cv2.cvtColor(predictions, cv2.COLOR_BGR2RGB) 


    #     predictions = resize(predictions, (256, 256))[..., :3]
    #     predictions = make_video(opt.swap_index, source_image, predictions, reconstruction_module, segmentation_module,
    #                             face_parser, hard=opt.hard, use_source_segmentation=opt.use_source_segmentation, cpu=opt.cpu) #Bug       
    #     # def video_swap(video_path, id_vetor,specific_person_id_nonorm,id_thres, swap_model, detect_model, save_path, temp_results_dir='./temp_results', crop_size=224, no_simswaplogo = False,use_mask =False):
    #     # print (type(swap_frame))
    #     # print (swap_frame.shape)
    #     # Display the swap frame
    #     #print("CAM")
    #     predictions = cv2.cvtColor(predictions, cv2.COLOR_BGR2RGB)
    # cv2.imshow('result_seg', predictions)
        



    for i in range (int(opt.num_seg)):
        # predictions = cv2.cvtColor(predictions, cv2.COLOR_BGR2RGB) 


        predictions = resize(predictions, (256, 256))[..., :3]
        predictions = make_video(opt.swap_index, source_image, predictions, reconstruction_module, segmentation_module,
                                face_parser, hard=opt.hard, use_source_segmentation=opt.use_source_segmentation, cpu=opt.cpu) #Bug       
        # def video_swap(video_path, id_vetor,specific_person_id_nonorm,id_thres, swap_model, detect_model, save_path, temp_results_dir='./temp_results', crop_size=224, no_simswaplogo = False,use_mask =False):
        # print (type(swap_frame))
        # print (swap_frame.shape)
        # Display the swap frame
        #print("CAM")
        # predictions = cv2.cvtColor(predictions, cv2.COLOR_BGR2RGB)
    # predictions = cv2.cvtColor(predictions, cv2.COLOR_BGR2RGB)

    print ('eww')
    print (predictions.shape)
    print (type(predictions))


    final_predictions = img_as_ubyte(predictions)

    return final_predictions
    # cv2.imshow('result_seg', predictions)
        # # Wait for key press
        # key = cv2.waitKey(1) & 0xFF
        
        # # Switch to the next webcam if 's' is pressed
        # if key == ord('s'):
        #     # Release the current webcam
        #     cap.release()
            
        #     # Increment the camera index
        #     current_camera_index = (current_camera_index + 1) % num_cameras
            
        #     # Create a VideoCapture object with the new webcam
        #     cap = cv2.VideoCapture(current_camera_index)
        
        # # Exit the loop if 'q' is pressed
        # if key == ord('q'):
        #     break

    # Release the webcam and close the OpenCV windows
    # cap.release()
    # cv2.destroyAllWindows()



    # Read fps of the target video and save result with the same fps
    # reader = imageio.get_reader(opt.target_video)
    # fps = reader.get_meta_data()['fps']
    # reader.close()

    # imageio.mimsave(opt.result_video, [img_as_ubyte(frame) for frame in predictions], fps=fps)
