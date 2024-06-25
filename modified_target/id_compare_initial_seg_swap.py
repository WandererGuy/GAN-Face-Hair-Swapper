import os 
import cv2
import torch
import numpy as np
from util.add_watermark import watermark_image
from util.norm import SpecificNorm
import torch.nn.functional as F
from parsing_model.model import BiSeNet
from modified_target.seg_swap_model import seg_swap_model
import time 

def _totensor(array):
    tensor = torch.from_numpy(array)
    img = tensor.transpose(0, 1).transpose(0, 2).contiguous()
    return img.float().div(255)

def find_min_index (detect_results, spNorm, swap_model, mse, specific_person_id_nonorm):
        frame_align_crop_list = detect_results[0]
        id_compare_values = [] 
        frame_align_crop_tenor_list = []
        for frame_align_crop in frame_align_crop_list:
            frame_align_crop_tenor = _totensor(cv2.cvtColor(frame_align_crop,cv2.COLOR_BGR2RGB))[None,...].cuda()
            frame_align_crop_tenor_arcnorm = spNorm(frame_align_crop_tenor)
            frame_align_crop_tenor_arcnorm_downsample = F.interpolate(frame_align_crop_tenor_arcnorm, size=(112,112))
            frame_align_crop_crop_id_nonorm = swap_model.netArc(frame_align_crop_tenor_arcnorm_downsample)

            id_compare_values.append(mse(frame_align_crop_crop_id_nonorm,specific_person_id_nonorm).detach().cpu().numpy())
            frame_align_crop_tenor_list.append(frame_align_crop_tenor)
        id_compare_values_array = np.array(id_compare_values)
        min_index = np.argmin(id_compare_values_array)
        min_value = id_compare_values_array[min_index]
        return min_index, min_value 



def seg_swap(whole_frame, opt, box_source,\
             specific_person_id_nonorm,id_thres, swap_model, detect_model, \
            seg_crop_size, no_simswaplogo ,use_mask, reconstruction_module, segmentation_module,face_parser, source, source_head_segmap, source_head_blend):
    logoclass = watermark_image('./simswaplogo/simswaplogo.png')
    spNorm =SpecificNorm()
    mse = torch.nn.MSELoss().cuda()
    frame = whole_frame
    min_index = 99
    if use_mask:
        n_classes = 19
        net = BiSeNet(n_classes=n_classes)
        net.cuda()
        save_pth = os.path.join('./parsing_model/checkpoint', '79999_iter.pth')
        net.load_state_dict(torch.load(save_pth))
        net.eval()
    else:
        net =None
    bbox_modify = opt.bbox_modify

    flag_face_shape = 0 

### CROP HEAD 
    # detect_results = [[generate_head_crop(frame)]]
    # cv2.imshow('head_target',(detect_results[0])[0])
    # cv2.waitKey(5000)
### detect_result is a numpy array of cropped image 


    detect_results_1 = detect_model.get(frame,seg_crop_size, 0)


    # FOR CUDAAAAAAAAAAAAAAAAAAAAAA , 3line 
    if opt.swap_option == 'swap_specific':
    #     if detect_results_1 is not None:
    #         min_index, min_value = find_min_index (detect_results_1, spNorm, swap_model, mse, specific_person_id_nonorm)
          
          
          
            # frame_align_crop_list = detect_results_1[0]
            # id_compare_values = [] 
            # frame_align_crop_tenor_list = []
            # for frame_align_crop in frame_align_crop_list:
            #     frame_align_crop_tenor = _totensor(cv2.cvtColor(frame_align_crop,cv2.COLOR_BGR2RGB))[None,...].cuda()
            #     frame_align_crop_tenor_arcnorm = spNorm(frame_align_crop_tenor)
            #     frame_align_crop_tenor_arcnorm_downsample = F.interpolate(frame_align_crop_tenor_arcnorm, size=(112,112))
            #     frame_align_crop_crop_id_nonorm = swap_model.netArc(frame_align_crop_tenor_arcnorm_downsample)

            #     id_compare_values.append(mse(frame_align_crop_crop_id_nonorm,specific_person_id_nonorm).detach().cpu().numpy())
            #     frame_align_crop_tenor_list.append(frame_align_crop_tenor)
            # id_compare_values_array = np.array(id_compare_values)
            # min_index = np.argmin(id_compare_values_array)
            # min_value = id_compare_values_array[min_index]
          


    ##### select closest id below threshold to decide swap bbox 
            # if min_value < id_thres:   
                    start_seg = time.time()
                    
                    min_index = 0 

                    ### temp 
                    # cv2.imshow('frame_align_crop_list[min_index]', frame_align_crop_list[min_index])
                    # cv2.waitKey(4000)
                    ### get a big head of target 
                    detect_results = detect_model.get(frame,seg_crop_size, bbox_modify)
                    # detect_results = detect_model.get(frame,seg_crop_size, 0)

                    frame_big_head_crop = (detect_results[0])[min_index]
                    frame_big_head_crop_mat = (detect_results[1])[min_index]

                    if opt.num_seg == 0 :
                            frame = frame_big_head_crop
                            frame_head_crop = frame 
                            swap_result = frame

                    else:

                            head_crop = time.time ()

                            # frame_head_crop, new_x_min, new_x_max, new_y_min, new_y_max  = generate_head_crop(frame_big_head_crop)

                            head_crop_end = time.time()
                            print ('headcrop {}'.format(head_crop_end-head_crop))

                            # if opt.show_intermediate_image == 1:
                            #     cv2.imshow('frame_app_big_head_crop',frame_big_head_crop)
                            #     cv2.waitKey(5000)

                            #     cv2.imshow('frame_seg_head_crop',frame_head_crop)
                            #     cv2.waitKey(5000)

                            cv2.imwrite('./ALL_TEST_IMAGE/3_frame_big_head_crop.jpg',frame_big_head_crop)

                            frame_big_head_crop = frame # just lazy , tthis should be commect out later
                            # frame_head_crop_detect_result = detect_model.get(frame_big_head_crop,seg_crop_size, 10) # this should be code 
                            frame_head_crop_detect_result = detect_model.get(frame_big_head_crop,seg_crop_size, bbox_modify)
                            # frame_head_crop_detect_result = detect_model.get(frame_big_head_crop,seg_crop_size, 0)

                            cv2.imwrite('./ALL_TEST_IMAGE/4_frame_head_crop_detect_result.jpg',frame_head_crop_detect_result[0][0])

        ############### make sure low sensitive det thres that detect many bbox , get the min_index bbox 
                            # second_min_index, _ = find_min_index (frame_head_crop_detect_result, spNorm, swap_model, mse, specific_person_id_nonorm)

                            second_min_index = 0 
                            frame_head_crop = (frame_head_crop_detect_result[0])[second_min_index]


                            SEG_model = time.time ()
                            swap_result = seg_swap_model (opt,source, source_head_segmap,source_head_blend, frame_head_crop, reconstruction_module, segmentation_module, face_parser, flag_face_shape)




        ##################### RETURN SMALL HEAD CROP TO BIG HEAD CROP IMAGE (NEED BIG HEAD CROP SO THAT APP.GET CAN DETECT LATER BETTER)
                            if opt.more_hair_rev == 1:
                                 hair_flag = 1
                            else :
                                 hair_flag = 0 
                            frame = reverse2original_image(opt, frame_head_crop_detect_result, 0, swap_result , frame_big_head_crop, seg_crop_size, hair_flag)
 
                            SEG_model_end = time.time()
                            print ('SEGGGG {}'.format(SEG_model_end-SEG_model))



                            end_seg = time.time()

                            print ('crop2ori {}'.format(end_seg - SEG_model_end))
                            print ('DONE SEG_SWAP')
                            print ('seg_swap : {}'.format(end_seg - start_seg))
                            print ('****************************')
                            # print (type(frame))
                            # print(frame)
                            # print(type(frame[0]))
                            # print(frame[0])
                            print(frame.shape)
                            cv2.imwrite('./ALL_TEST_IMAGE/5_seg_swap_result.jpg',frame)
                            print ('DONE SEG_SWAP')
                            print ('seg_swap : {}'.format(end_seg - start_seg))
        #     else:
        #             min_index = 99 
        #             detect_results = 0 
        #             frame = frame.astype(np.uint8)
        #             if not no_simswaplogo:
        #                 frame = logoclass.apply_frames(frame)
        # else:
        #         min_index = 99 
        #         detect_results = 0    
        #         frame = frame.astype(np.uint8)
        #         if not no_simswaplogo:
        #             frame = logoclass.apply_frames(frame)






    # cv2.imshow('whole_frame', whole_frame )
    # cv2.waitKey(4000)
    # cv2.imshow('frame', frame )
    # cv2.waitKey(4000)

    # if opt.return_grid == 1: 
    return frame, min_index, detect_results
    # else :
    #     return frame, min_index, frame_big_head_crop, spNorm, mse, frame_big_head_crop_mat





def reverse2original_image(opt, detect_results, min_index, swapped_image , whole_frame, seg_crop_size, hair_flag):
        from util.reverse2original import reverse2wholeimage

        from util.norm import SpecificNorm
        from util.add_watermark import watermark_image

        spNorm =SpecificNorm()
        logoclass = watermark_image('./simswaplogo/simswaplogo.png')

        if opt.use_mask:
            n_classes = 19
            net = BiSeNet(n_classes=n_classes)
            net.cuda()
            save_pth = os.path.join('./parsing_model/checkpoint', '79999_iter.pth')
            net.load_state_dict(torch.load(save_pth))
            net.eval()
        else:
            net =None



        img_c_align_crop_list, c_mat_list = detect_results


        start_rev2 = time.time ()
        # detect_results = None
        swap_result_list = []

        c_align_crop_tenor_list = []

        for c_align_crop in img_c_align_crop_list:

            c_align_crop_tenor = _totensor(cv2.cvtColor(c_align_crop,cv2.COLOR_BGR2RGB))[None,...].cuda()

            # swap_result = model(None, b_align_crop_tenor, latend_id, None, True)[0]

            rgb_image = cv2.cvtColor(swapped_image, cv2.COLOR_BGR2RGB) 

            start = time.time()
            # Min-Max scaling
            # min_val = np.min(rgb_image)
            # max_val = np.max(rgb_image)
            # norm_image = (rgb_image - min_val) / (max_val - min_val)
            norm_image = rgb_image/255

            # since output of simswap model standard is tensor (3,h,w) and normalized for input to rev2original and in BGR 
            swap_result = torch.from_numpy(norm_image.transpose(2,0,1))
            swap_result_list.append(swap_result)
            c_align_crop_tenor_list.append(c_align_crop_tenor)

            end = time.time()
            print ('pre ori 2 {}'.format(end-start))
        image = reverse2wholeimage([c_align_crop_tenor_list[min_index]], swap_result_list, [c_mat_list[min_index]], seg_crop_size, whole_frame, logoclass, \
                os.path.join(opt.output_path, 'result_whole_swapsingle.jpg'), opt.no_simswaplogo,pasring_model =net, norm = spNorm, use_mask=opt.use_mask)
        
        

        return image
