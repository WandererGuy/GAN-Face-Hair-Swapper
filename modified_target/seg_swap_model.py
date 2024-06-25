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


import time 

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
        config = yaml.load(f, Loader=yaml.Loader)

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


def make_video(swap_index, source, source_head_segmap, source_head_blend, target_video, reconstruction_module, segmentation_module, face_parser=None,
               hard=False, use_source_segmentation=False, cpu=False):
    assert type(swap_index) == list
    with torch.no_grad():



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





        out = reconstruction_module(source, target_frame, seg_source=source_head_segmap, seg_target=seg_target,
                                    blend_mask=source_head_blend, use_source_segmentation=use_source_segmentation)

        # predictions.append(np.transpose(out['prediction'].data.cpu().numpy(), [0, 2, 3, 1])[0])
        predictions = np.transpose(out['prediction'].data.cpu().numpy(), [0, 2, 3, 1])[0]


    return predictions




def seg_swap_model(opt, source, source_head_segmap,source_head_blend, box_target, reconstruction_module, segmentation_module, face_parser, flag_face_shape):


    target_video = box_target 
    

    # blend_scale = (512 / 4) / 512 if opt.supervised else 1
    # reconstruction_module, segmentation_module = load_checkpoints(opt.config, opt.checkpoint, blend_scale=blend_scale, 
    #                                                               first_order_motion_model=opt.first_order_motion_model, cpu=opt.cpu)

    # if opt.supervised:
    #     face_parser = load_face_parser(opt.cpu)
    # else:
    #     face_parser = None


    target_video = resize(target_video, (256, 256))[..., :3]
    
    # predictions = target_video
    # print (target_video.shape)
    # print (type(target_video))


    predictions = target_video

 
    for i in range (int(opt.num_seg)):
        predictions = resize(predictions, (256, 256))[..., :3]
        # predictions = resize(predictions, (224, 224))[..., :3]
        if flag_face_shape == 0:
            predictions = make_video(opt.swap_index, source, source_head_segmap, source_head_blend, predictions, reconstruction_module, segmentation_module,
                                        face_parser, hard=opt.hard, use_source_segmentation=opt.use_source_segmentation, cpu=opt.cpu) #Bug     
        if flag_face_shape == 1:
            predictions = make_video([1], source, source_head_segmap, source_head_blend, predictions, reconstruction_module, segmentation_module,
                                        face_parser, hard=opt.hard, use_source_segmentation=opt.use_source_segmentation, cpu=opt.cpu) #Bug     

    final_predictions = img_as_ubyte(predictions)
    # print('OMGAGAGAGAGA')
    # print (final_predictions.shape)

    return final_predictions
 