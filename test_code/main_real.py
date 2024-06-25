# from seg_swap.swap_part import PartSwapGenerator, load_checkpoints, load_face_parser, make_video, seg_swap_model
from modified_target.id_compare_initial_seg_swap import seg_swap
from modified_target.seg_option import SegOptions
from modified_target.bbox_preprocess import preprocess_segment
import cv2

from webcam import turn_on_webcam


if __name__ == "__main__":

    opt = SegOptions().parse()
    # read image , create bbox , extract id
    # god = cv2.imread('/home/manh/Downloads/bean.jpg')
    # cv2.imshow('test', god)

    pic_b = turn_on_webcam()


    img_a_align_crop, latend_id,specific_person_id_nonorm, opt.id_thres, \
            model, app,output_path,no_simswaplogo,use_mask,crop_size = preprocess_segment(opt, pic_b)


    #  return img_a_align_crop, latend_id,specific_person_id_nonorm, opt.id_thres, \
    #        model, app,opt.output_path,no_simswaplogo=opt.no_simswaplogo,use_mask=opt.use_mask,crop_size=crop_size


    # input pic -> get segment result ->  reverse to whole image -> display 
#     result = seg_swap(opt.supervised, opt.config, opt.checkpoint, opt.cpu, opt.num_seg, opt.swap_index, opt.hard, opt.use_source_segmentation, img_a_align_crop, opt.whole_frame, latend_id,specific_person_id_nonorm, opt.id_thres, \
#             model, app,output_path,no_simswaplogo,use_mask,crop_size)



    # result = seg_swap(opt, img_a_align_crop[0], opt.frame_path, latend_id,specific_person_id_nonorm, opt.id_thres, \
    #         model, app,output_path,crop_size, no_simswaplogo,use_mask)








    cap = cv2.VideoCapture(0)
    while True:
        # Capture frame-by-frame
        ret, frame = cap.read()
        
        # Check if the frame is successfully captured
        if not ret:
            print("Failed to capture frame")
            break
        
        result = seg_swap(frame, opt, img_a_align_crop[0], opt.frame_path, latend_id,specific_person_id_nonorm, opt.id_thres, \
        model, app,output_path,crop_size, no_simswaplogo,use_mask)

        # Display the resulting frame
        cv2.imshow('bbox_Webcam', result)
        key = cv2.waitKey(300)
        print ('1')




    print ('noice_done')
    print (opt.output_path)
    cv2.imwrite (opt.output_path, result )

    # display_grid()
