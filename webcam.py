import cv2
import torch 
from sub_process_video import warm_up, make_image
import time 

def swap_pipeline(frame, opt,img_a_whole, img_a_align_crop, \
            specific_person_id_nonorm, model, app, \
            crop_size, no_simswaplogo,use_mask, reconstruction_module, segmentation_module, face_parser, latend_id, source_head_crop):
        
        new_frame = make_image (frame, opt,img_a_whole, img_a_align_crop, \
            specific_person_id_nonorm, model, app, \
            crop_size, no_simswaplogo,use_mask, reconstruction_module, segmentation_module, face_parser, latend_id, source_head_crop)
        return new_frame

def generate_gan(opt):
        ### warm_up : prepare variable value and generate gan face for specific image 
        gan_img, img_a_whole, img_a_align_crop, \
        specific_person_id_nonorm, \
        model, app, output_path,no_simswaplogo,use_mask,crop_size, reconstruction_module, segmentation_module, face_parser, latend_id, source_head_crop = warm_up(opt)
        
        return gan_img, img_a_whole, img_a_align_crop, \
        specific_person_id_nonorm, \
        model, app, output_path,no_simswaplogo,use_mask,crop_size, reconstruction_module, segmentation_module, face_parser, latend_id, source_head_crop
    
def swap_webcam(opt):
        start1 = time.time()
        frame_index = 0 
        # Open the default camera (index 0)
        gan_img, img_a_whole, img_a_align_crop, \
        specific_person_id_nonorm, \
        model, app, output_path,no_simswaplogo,use_mask,crop_size, reconstruction_module, segmentation_module, face_parser, latend_id, source_head_crop = generate_gan(opt)
        print ('DONE GENERATE GAN FACE ')
        end1 = time.time()
        generate_time = end1-start1
        print ('GENERATE GAN: {} '.format(generate_time))

        
        cap = cv2.VideoCapture(0)
        
        # Check if the webcam is successfully opened
        if not cap.isOpened():
            print("Failed to open webcam")
            return
        
        # Read and display frames from the webcam
        while True:
            # Capture frame-by-frame
            ret, frame = cap.read()
            
            # Check if the frame is successfully captured
            if not ret:
                print("Failed to capture frame")
                break
            start2 = time.time()


            frame_index = frame_index + 1
            swap_frame = swap_pipeline(frame, opt,img_a_whole, img_a_align_crop, \
            specific_person_id_nonorm, model, app, \
            crop_size, no_simswaplogo,use_mask, reconstruction_module, segmentation_module, face_parser, latend_id, source_head_crop)
        
            end2 = time.time()
            pipeline_time = end2-start2
            print ('PIPELINE: {} '.format(pipeline_time))
            # Display the resulting frame
            
            cv2.imshow('frame', swap_frame)
            
            print ('*******************DONE FRAME: {}*******************'.format(frame_index))
            cv2.waitKey(200)
            torch.cuda.empty_cache() # empty cache 
        cap.release()
        cv2.destroyAllWindows()
        



