import os 
import cv2
import glob
import torch
import shutil
import numpy as np
from tqdm import tqdm
from util.reverse2original import reverse2wholeimage
import moviepy.editor as mp
from moviepy.editor import AudioFileClip, VideoFileClip 
from moviepy.video.io.ImageSequenceClip import ImageSequenceClip
import  time
from util.add_watermark import watermark_image
from util.norm import SpecificNorm
import torch.nn.functional as F
from parsing_model.model import BiSeNet

def _totensor(array):
    tensor = torch.from_numpy(array)
    img = tensor.transpose(0, 1).transpose(0, 2).contiguous()
    return img.float().div(255)


def video_swap(ret, frame, id_vetor,specific_person_id_nonorm,id_thres, swap_model, detect_model, save_path, temp_results_dir='./temp_results', crop_size=224, no_simswaplogo = False,use_mask =False):
    logoclass = watermark_image('./simswaplogo/simswaplogo.png')
    spNorm =SpecificNorm()
    mse = torch.nn.MSELoss().cuda()

    if use_mask:
        n_classes = 19
        net = BiSeNet(n_classes=n_classes)
        net.cuda()
        save_pth = os.path.join('./parsing_model/checkpoint', '79999_iter.pth')
        net.load_state_dict(torch.load(save_pth))
        net.eval()
    else:
        net =None

    if  ret:
        detect_results = detect_model.get(frame,crop_size)

        if detect_results is not None:
            # print(frame_index)
            # if not os.path.exists(temp_results_dir):
            #         os.mkdir(temp_results_dir)
            frame_align_crop_list = detect_results[0]
            frame_mat_list = detect_results[1]

            id_compare_values = [] 
            frame_align_crop_tenor_list = []
            for frame_align_crop in frame_align_crop_list:

                # BGR TO RGB
                # frame_align_crop_RGB = frame_align_crop[...,::-1]

                frame_align_crop_tenor = _totensor(cv2.cvtColor(frame_align_crop,cv2.COLOR_BGR2RGB))[None,...].cuda()
                frame_align_crop_tenor_arcnorm = spNorm(frame_align_crop_tenor)
                frame_align_crop_tenor_arcnorm_downsample = F.interpolate(frame_align_crop_tenor_arcnorm, size=(112,112))
                frame_align_crop_crop_id_nonorm = swap_model.netArc(frame_align_crop_tenor_arcnorm_downsample)

                id_compare_values.append(mse(frame_align_crop_crop_id_nonorm,specific_person_id_nonorm).detach().cpu().numpy())
                frame_align_crop_tenor_list.append(frame_align_crop_tenor)
            id_compare_values_array = np.array(id_compare_values)
            min_index = np.argmin(id_compare_values_array)
            min_value = id_compare_values_array[min_index]
            if min_value < id_thres:   

                print ('okla')

                swap_result = swap_model(None, frame_align_crop_tenor_list[min_index], id_vetor, None, True)[0]
                # reverse2original seems consuming , comsider optimize 
                # reverse2wholeimage([frame_align_crop_tenor_list[min_index]], [swap_result], [frame_mat_list[min_index]], crop_size, frame, logoclass,\
                #     os.path.join(temp_results_dir, 'frame_{:0>7d}.jpg'.format(frame_index)),no_simswaplogo,pasring_model =net,use_mask= use_mask, norm = spNorm)
                print (type(swap_result))
                

                # def reverse2wholeimage(b_align_crop_tenor_list,swaped_imgs, mats, crop_size, oriimg, logoclass, save_path = '', \
                #                     no_simswaplogo = False,pasring_model =None,norm = None, use_mask = False):

                # reverse2wholeimage([frame_align_crop_tenor_list[min_index]], [swap_result], [frame_mat_list[min_index]], crop_size, frame, logoclass,\
                #       os.path.join(temp_results_dir, 'frame_{:0>7d}.jpg'.format(frame_index)),no_simswaplogo,pasring_model =net,use_mask= use_mask, norm = spNorm)
                
                print ('okkkk')


                frame = reverse2wholeimage([frame_align_crop_tenor_list[min_index]], [swap_result], [frame_mat_list[min_index]], crop_size, frame, logoclass,\
                        no_simswaplogo,pasring_model =net,use_mask= use_mask, norm = spNorm)
                
                # add return final_img in util.reverse2original.reverse2wholeimage
                # since author only write result in save path , no return of result frame 
                # above func is process for img
            else:
                frame = frame.astype(np.uint8)
                if not no_simswaplogo:
                    frame = logoclass.apply_frames(frame)
                # if not os.path.exists(temp_results_dir):
                #     os.mkdir(temp_results_dir)

                # cv2.imwrite(os.path.join(temp_results_dir, 'frame_{:0>7d}.jpg'.format(frame_index)), frame)

        else:
            # if not os.path.exists(temp_results_dir):
            #     os.mkdir(temp_results_dir)
            frame = frame.astype(np.uint8)
            if not no_simswaplogo:
                frame = logoclass.apply_frames(frame)
            # cv2.imwrite(os.path.join(temp_results_dir, 'frame_{:0>7d}.jpg'.format(frame_index)), frame)
    else:
        return 
    print("done")
    return frame 