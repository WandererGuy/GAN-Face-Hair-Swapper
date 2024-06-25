import os 
import torch
from tqdm import tqdm
import cv2
from modified_target.seg_option import SegOptions
from sub_process_video import warm_up, make_image

if __name__ == '__main__':
    torch.cuda.empty_cache()       
    opt = SegOptions().parse()    
### initiate value 
    # pic_specific = cv2.imread (opt.pic_specific_path) 
    temp_results_dir = opt.temp_results_dir
    ret = True
    frame_index = 0
    video = cv2.VideoCapture(opt.the_video_path)
    frame_count = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = video.get(cv2.CAP_PROP_FPS)

    for frame_index in range(1): 
        first_img = (video.read())[1]
    pic_specific = first_img


### warm_up : prepare variable value and generate gan face for specific image 

    gan_img, img_a_whole, img_a_align_crop, \
        specific_person_id_nonorm, \
        model, app, output_path,no_simswaplogo,use_mask,crop_size, reconstruction_module, segmentation_module, face_parser, latend_id, source_head_crop = warm_up(opt, pic_specific)
    
### swap each frame and save output 
    for frame_index in tqdm(range(frame_count)): 
        ret, frame = video.read()
        if  ret:
            new_frame = make_image (frame, opt,img_a_whole, img_a_align_crop, \
            specific_person_id_nonorm, model, app, \
            crop_size, no_simswaplogo,use_mask, reconstruction_module, segmentation_module, face_parser, latend_id, source_head_crop)
            torch.cuda.empty_cache() # empty cache 
            cv2.imwrite(os.path.join(temp_results_dir, 'frame_{:0>7d}.jpg'.format(frame_index)), new_frame)

        else:
            video.release()







