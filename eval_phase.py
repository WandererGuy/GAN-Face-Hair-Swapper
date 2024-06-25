# from seg_swap.swap_part import PartSwapGenerator, load_checkpoints, load_face_parser, make_video, seg_swap_model
from modified_target.id_compare_initial_seg_swap import seg_swap
from modified_target.seg_option import SegOptions
from modified_target.bbox_preprocess import preprocess_segment
import cv2

from webcam import turn_on_webcam
from file_test_wholeimage_swapsingle import test_wholeimage_swapsingle
from util.file_test_wholeimage_swapsingle_2 import test_wholeimage_swapsingle_2

from seg_swap_2 import seg_swap_2


import matplotlib.pyplot as plt
import math
import os
import numpy as np
from show_fake_face import generate 
import torch 
# def grid_show():



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
    fig.savefig('/home/manh/coding/faceswap/output/fullY30.png')

    print ('11111111111111')





# if __name__ == "__main__":
def two_phases(opt):
    opt = SegOptions().parse()
    # read image , create bbox , extract id
    # god = cv2.imread('/home/manh/Downloads/bean.jpg')
    # cv2.imshow('test', god)
    gan_img = None 

    pic_specific = cv2.imread (opt.pic_specific_path)  ### for ref id 
    whole_frame = cv2.imread (opt.frame_path)  ### frame that have person to swap 
    min_index = 99

    if opt.gan_face:
        gan_img_pillow = generate(pic_specific)
        im_pillow = np.array(gan_img_pillow) # turn to array 
        gan_img = cv2.cvtColor(im_pillow, cv2.COLOR_RGB2BGR) 
        # gan_img = cv2.resize (gan_img, (256, 256))
    torch.cuda.empty_cache()
        
    print (opt.visualize)
    if opt.visualize == 'True' and opt.gan_face == 0:
        from visualize import visualize_segmentation
        import imageio
        source_image = imageio.imread(opt.source_image)

        visualize_segmentation(source_image)






    print (opt.use_mask_face_part_id_list)
    print (type([opt.use_mask_face_part_id_list]))

    print (opt.swap_index)
    print (type([opt.swap_index]))
    img_a_whole, img_a_align_crop, latend_id,specific_person_id_nonorm, opt.id_thres, \
            model, app,output_path,no_simswaplogo,use_mask,crop_size, img_a_align_crop_2 = preprocess_segment(opt, pic_specific, gan_img)
    
    # cv2.imshow ('source_align_crop', img_a_align_crop[0])

    seg_result, min_index = seg_swap(whole_frame, opt, img_a_align_crop[0], opt.frame_path, latend_id,specific_person_id_nonorm, opt.id_thres, \
    model, app,output_path,crop_size, no_simswaplogo,use_mask)

    if opt.visualize == 'True' and opt.gan_face == 1:
        from visualize import visualize_segmentation
        visualize_segmentation(img_a_align_crop[0] , bbox_frame)

    print ('KKKKKKKKKKK')
    print (min_index)

    if min_index != 99 or opt.swap_option == 'swap_single':
        if opt.face_shape:
            seg_result_2, crop_frame_2 = seg_swap_2(seg_result, opt, img_a_align_crop_2[0], opt.frame_path, latend_id,specific_person_id_nonorm, opt.id_thres, \
            model, app,output_path,crop_size, no_simswaplogo,use_mask)


            




    # #### new 
    #     seg_result, abc = seg_swap_2(img_a_align_crop[0], opt, seg_result, opt.frame_path, latend_id,specific_person_id_nonorm, opt.id_thres, \
    #     model, app,output_path,crop_size, no_simswaplogo,use_mask)

    #     seg_result, crop_frame = seg_swap_2(crop_frame, opt, seg_result, opt.frame_path, latend_id,specific_person_id_nonorm, opt.id_thres, \
    #     model, app,output_path,crop_size, no_simswaplogo,use_mask)






        # cv2.imshow('seg_result', seg_result)
        # cv2.waitKey(5000)    




        # cv2.imshow('seg_result', seg_result)
        # cv2.waitKey(10000)

        if opt.face_shape:
            from reverse_back import reverse_back
            seg_result_new =  reverse_back(opt, seg_result, seg_result_2, app)

            simswap_result, final_img = test_wholeimage_swapsingle(opt, img_a_whole, seg_result_new, app, whole_frame, min_index)
            list_img = [img_a_whole, whole_frame, img_a_align_crop[0], crop_frame, seg_result, seg_result_2,seg_result_new, simswap_result, final_img]

            for i in list_img:
                cv2.imshow('webcam', i)
                cv2.waitKey(3500)
            # list_img = [img_a_whole, whole_frame, img_a_align_crop[0], crop_frame, seg_result, simswap_result, simswap_result_2]
            images_name = ['source_whole', 'whole_frame_target', 'source_align_crop', 'frame_align_crop', 'segmnent_result','seg_result_2_face_shape','seg_result_new', 'simswap_result', 'final_img']

            grid_size = (5, 5)
            view_images_in_grid(list_img, grid_size, images_name)
            print ('OVERRRRRRRRRRRRRRRRR')


        else:    
            simswap_result, final_img = test_wholeimage_swapsingle(opt, img_a_whole, seg_result, app, whole_frame, min_index)

        return final_img
        # cv2.imshow('simswap_result', simswap_result)

        # simswap_result_2 = test_wholeimage_swapsingle_2(opt, img_a_whole, simswap_result, app, whole_frame)
        
        # cv2.waitKey(5000)
''''''
'''    list_img = [img_a_whole, whole_frame, img_a_align_crop[0], seg_result, simswap_result, final_img]

            for i in list_img:
                new_i= (i-np.min(i))/(np.max(i)-np.min(i))

                cv2.imshow('webcam', new_i)
                cv2.waitKey(3500)
            # list_img = [img_a_whole, whole_frame, img_a_align_crop[0], crop_frame, seg_result, simswap_result, simswap_result_2]
            images_name = ['source_whole', 'whole_frame_target', 'source_align_crop', 'segmnent_result', 'simswap_result', 'final_img']

            grid_size = (5, 5)
            view_images_in_grid(list_img, grid_size, images_name)
            print ('OVERRRRRRRRRRRRRRRRR')
    
    else:
        final_img = whole_frame 
        cv2.imshow ('webcam', final_img)

'''


        # whole_frame
        # img_a_align_crop[0]
        # crop_frame
        # seg_result
        # simswap_result

        # cap = cv2.VideoCapture(0)
        # while True:
        #     # Capture frame-by-frame
        #     ret, frame = cap.read()
            
        #     # Check if the frame is successfully captured
        #     if not ret:
        #         print("Failed to capture frame")
        #         break
            

            # key = cv2.waitKey(100)




        # print ('noice_done')
        # print (opt.output_path)
        # cv2.imwrite (opt.output_path, result )

        # display_grid()