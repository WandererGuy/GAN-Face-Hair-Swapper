# from seg_swap.swap_part import PartSwapGenerator, load_checkpoints, load_face_parser, make_video, seg_swap_model
from modified_target.id_compare_initial_seg_swap import seg_swap
from modified_target.seg_option import SegOptions
from modified_target.bbox_preprocess import preprocess_segment
import cv2

if __name__ == "__main__":

    opt = SegOptions().parse()
    # preprocess input + bounding box + select box (identity) to swap 
    img_a_align_crop, latend_id,specific_person_id_nonorm, opt.id_thres, \
            model, app,output_path,no_simswaplogo,use_mask,crop_size = preprocess_segment(opt)


    #  return img_a_align_crop, latend_id,specific_person_id_nonorm, opt.id_thres, \
    #        model, app,opt.output_path,no_simswaplogo=opt.no_simswaplogo,use_mask=opt.use_mask,crop_size=crop_size


    # input pic -> get segment result ->  reverse to whole image -> display 
#     result = seg_swap(opt.supervised, opt.config, opt.checkpoint, opt.cpu, opt.num_seg, opt.swap_index, opt.hard, opt.use_source_segmentation, img_a_align_crop, opt.whole_frame, latend_id,specific_person_id_nonorm, opt.id_thres, \
#             model, app,output_path,no_simswaplogo,use_mask,crop_size)
    result = seg_swap(opt, img_a_align_crop[0], opt.frame_path, latend_id,specific_person_id_nonorm, opt.id_thres, \
            model, app,output_path,crop_size, no_simswaplogo,use_mask)

# def seg_swap(opt, box_source, frame_path, latend_id ,specific_person_id_nonorm,id_thres, swap_model, detect_model, save_path, crop_size, no_simswaplogo ,use_mask ):


    print ('noice_done')

    

    # display_grid()

