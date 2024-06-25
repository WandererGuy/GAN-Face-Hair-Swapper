from modified_target.id_compare_initial_seg_swap import seg_swap
from modified_target.seg_option import SegOptions
from modified_target.bbox_preprocess import preprocess_segment
import cv2
# from webcam import turn_on_webcam
from file_test_wholeimage_swapsingle import test_wholeimage_swapsingle
import matplotlib.pyplot as plt
import math
import numpy as np
from show_fake_face import generate 
import torch 
import time 
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
from skimage.transform import resize



from modified_target.seg_swap_model import load_checkpoints, load_face_parser

### VIEW IMAGE IN GRID 
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


### GENERATE GAN CORRESPOND TO SPECIFIC PIC 
def warm_up(opt, pic_specific):

        # min_index = 99
        if opt.gan_face:
                gan_img_pillow = generate(pic_specific)
                im_pillow = np.array(gan_img_pillow) # turn to array 
                gan_img = cv2.cvtColor(im_pillow, cv2.COLOR_RGB2BGR) 
                cv2.imwrite(opt.save_gan_image_path, gan_img)
                print ('DONE GENERATE GAN FACE')

        if opt.gan_face == 0:
               gan_img = cv2.imread (opt.source_image)


        torch.cuda.empty_cache()
        
        img_a_whole,specific_person_id_nonorm, opt.id_thres, \
                model, app,output_path,no_simswaplogo,use_mask,crop_size = preprocess_segment(opt, pic_specific, gan_img)
        print ('DONE EXTRACT SPECIFIC FACE LATENT ID')

################## initiate values for seg swap model ############


        blend_scale = (256 / 4) / 512 if opt.supervised else 1
        reconstruction_module, segmentation_module = load_checkpoints(opt.config, opt.checkpoint, blend_scale=blend_scale, 
                                                                  first_order_motion_model=opt.first_order_motion_model, cpu=opt.cpu)
        # segmentation_module = segmentation_module.cuda()
        # reconstruction_module = reconstruction_module.cuda()
        if opt.supervised:
                # face_parser = load_face_parser(opt.cpu)
                face_parser = load_face_parser(opt.cpu)

        else:
                face_parser = None

        
################### latent id for simswap  ####################


        transformer_Arcface = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        img_a_align_crop, _ = app.get(img_a_whole,crop_size, 0)

        img_a_align_crop_pil = Image.fromarray(cv2.cvtColor(img_a_align_crop[0],cv2.COLOR_BGR2RGB)) 
        img_a = transformer_Arcface(img_a_align_crop_pil)
        img_id = img_a.view(-1, img_a.shape[0], img_a.shape[1], img_a.shape[2])

        # convert numpy to tensor
        img_id = img_id.cuda()

        #create latent id
        img_id_downsample = F.interpolate(img_id, size=(112,112))
        latend_id = model.netArc(img_id_downsample)
        latend_id = F.normalize(latend_id, p=2, dim=1)

        if opt.gan_face == 0:
                   source_head_crop_result,_ = app.get (img_a_whole, crop_size, 15)
                   source_head_crop = source_head_crop_result[0]

        if opt.gan_face == 1:
                        source_head_crop = img_a_whole


        print ('DONE EXTRACT SOURCE GAN FACE LATENT ID ')
        
        source_image = img_a_whole #### this can be fix to = source_head_crop for more flexibility 
        source_image = resize(source_image, (256, 256))[..., :3]
        
        
        source = torch.tensor(source_image[np.newaxis].astype(np.float32)).permute(0, 3, 1, 2)
# if not cpu:
        source = source.cuda()
        # print("Oke")
        source_head_segmap = segmentation_module(source)
        with torch.no_grad():
                if face_parser is not None:
                        blend_mask = F.interpolate(source, size=(512, 512))
                        # face_parser.mean = face_parser.mean.cuda()
                        # face_parser.std = face_parser.std.cuda()
                        # face_parser = face_parser.cuda()
                        
                        blend_mask = (blend_mask - face_parser.mean) / face_parser.std
                        blend_mask = torch.softmax(face_parser(blend_mask)[0], dim=1)
                else:
                        blend_mask = source_head_segmap['segmentation']  

                        
                source_head_blend = blend_mask[:, opt.swap_index].sum(dim=1, keepdim=True)

                # if opt.hard:
                #         source_head_blend = (blend_mask > 0.5).type(blend_mask.type())
        cv2.imwrite('./ALL_TEST_IMAGE/1_source_image.jpg', img_a_whole)


        print ('DONE SEGMENTATION and BLEND MASK ON SOURCE')
        return gan_img, img_a_whole, img_a_align_crop,specific_person_id_nonorm, \
                model, app,output_path,no_simswaplogo,use_mask,crop_size, reconstruction_module, segmentation_module, face_parser, latend_id, source, source_head_segmap, source_head_blend


### SEGMENT AND SWAP 
def make_image (whole_frame, opt,img_a_whole, img_a_align_crop ,specific_person_id_nonorm, \
                    model, app,crop_size, no_simswaplogo,use_mask, reconstruction_module, segmentation_module, face_parser, latend_id, source, source_head_segmap, source_head_blend):
        ### SEGMENT SWAP ON MIN_INDEX IDENTITY 
        # start_extract_seg = time.time()
        all_swap_start = time.time()
        print ('START SEG_SWAP...')
        seg_result, min_index, detect_results = seg_swap(whole_frame, opt, img_a_align_crop[0],specific_person_id_nonorm, opt.id_thres, \
        model, app,crop_size, no_simswaplogo,use_mask, reconstruction_module, segmentation_module, face_parser,source,  source_head_segmap, source_head_blend)
        

        # end_extract_seg = time.time()
        # print ('DONE EXTRACT + SEGMENT SWAP')
        # print ('EXTRACT + SEG_SWAP : {}'.format(end_extract_seg - start_extract_seg))

        ### CROP FROM SEG_RESULT AND SWAPPING 
        torch.cuda.empty_cache()       

        if min_index != 99 or opt.swap_option == 'swap_single':
                print ('START FACESWAP...')
                final_img = test_wholeimage_swapsingle(opt, img_a_whole, seg_result, whole_frame, min_index, detect_results, whole_frame, latend_id, app, model)


                all_swap_end = time.time()
                print ('ALL_SWAP : {}'.format(all_swap_end-all_swap_start))

                print ('*********SWAPPED*********')


    ### 99 means detect no id get extracted -> means no face detected in camera -> BLACK SCREEN
        elif min_index == 99 :
                # height = whole_frame.shape[0]
                # width = whole_frame.shape[1]
                # dark_image = np.zeros((height, width, 3), dtype=np.uint8)
                # final_img = dark_image 
                final_img = whole_frame
                print ('*********NO_SWAP*********')
        return final_img
