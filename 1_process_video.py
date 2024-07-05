from tqdm import tqdm
import cv2
from modified_target.seg_option import SegOptions
from sub_process_video import warm_up, make_image
import matplotlib.pyplot as plt
import math
import os 
import torch
def view_images_in_grid(images, grid_size, images_name):
    num_images = len(images)
    rows = math.ceil(num_images / grid_size[1])
    fig, axes = plt.subplots(rows, grid_size[1], figsize=(12, 8))

    for i, ax in enumerate(axes.flat):
        if i < num_images:
            # img = plt.imread(image_files[i])
            img_bgr = images[i]
            img = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

            ax.imshow(img)
            ax.axis('off')
            ax.set_title(images_name[i], fontsize=12)
        else:
            ax.axis('off')

    plt.tight_layout()
    plt.show()
    fig.savefig(opt.save_grid_path)

if __name__ == '__main__':
    print ('STARTING...')
    if not os.path.exists('./ALL_TEST_IMAGE'):
        os.mkdir('./ALL_TEST_IMAGE')
    torch.cuda.empty_cache()       
    opt = SegOptions().parse()    
    specific_image = cv2.imread(opt.pic_specific_path)
### warm_up : prepare variable value and generate gan face for specific image 
    gan_img, img_a_whole, img_a_align_crop, \
        specific_person_id_nonorm, \
        model, app, output_path,no_simswaplogo,use_mask,crop_size, reconstruction_module, segmentation_module, face_parser, latend_id, source, source_head_segmap, source_head_blend = warm_up(opt, specific_image)
    
    print ('DONE PHASE 1')
    torch.cuda.empty_cache()       

### swap frame and save output 
    os.makedirs('temp', exist_ok=True)

    temp_results_dir = opt.temp_results_dir
    ret = True
    frame_index = 0
    video = cv2.VideoCapture(opt.the_video_path)
    frame_count = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = video.get(cv2.CAP_PROP_FPS)

    cv2.imwrite('./ALL_TEST_IMAGE/2_target.jpg', frame)

### swap each frame and save output 
    for frame_index in tqdm(range(frame_count)): 
        ret, frame = video.read()
        if  ret:
            new_frame = make_image (frame, opt,img_a_whole, img_a_align_crop, \
            specific_person_id_nonorm, model, app, \
            crop_size, no_simswaplogo,use_mask, reconstruction_module, segmentation_module, face_parser, latend_id, source, source_head_segmap, source_head_blend)
            torch.cuda.empty_cache() # empty cache 
            cv2.imwrite(os.path.join(temp_results_dir, 'frame_{:0>7d}.jpg'.format(frame_index)), new_frame)

        else:
            video.release()







