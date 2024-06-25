import cv2
import torch
import fractions
import numpy as np
import torch.nn.functional as F
from models.models import create_model
from util.reverse2original import reverse2wholeimage
import os
from util.add_watermark import watermark_image
from util.norm import SpecificNorm
from parsing_model.model import BiSeNet
import time
from insightface_func.face_detect_crop_multi import Face_detect_crop


def lcm(a, b): return abs(a * b) / fractions.gcd(a, b) if a and b else 0



def _totensor(array):
    tensor = torch.from_numpy(array)
    img = tensor.transpose(0, 1).transpose(0, 2).contiguous()
    return img.float().div(255)


def test_wholeimage_swapsingle(opt, img_a_whole, result, frame_whole, min_index, detect_results, whole_frame, latend_id, app, model):


    start_simswap = time.time()

    crop_size = opt.crop_size

    # torch.nn.Module.dump_patches = True

    # if crop_size == 512:
    #     opt.which_epoch = 550000
    #     opt.name = '512'
    #     mode = 'ffhq'
    # else:
    #     mode = 'None'
    
    app.prepare(ctx_id= 0, det_thresh=0.3, det_size=(640,640),mode='None')

    logoclass = watermark_image('./simswaplogo/simswaplogo.png')

    
    # model = create_model(opt)
    # model.eval()


    prepare_input_MODEL = time.time()

    spNorm =SpecificNorm()

    if opt.use_mask:
        n_classes = 19
        net = BiSeNet(n_classes=n_classes)
        net.cuda()
        save_pth = os.path.join('./parsing_model/checkpoint', '79999_iter.pth')
        net.load_state_dict(torch.load(save_pth))
        net.eval()
    else:
        net =None


    with torch.no_grad():
        prepare_input_MODEL = time.time()
        img_b_whole = result

        img_b_align_crop_list, b_mat_list = app.get(img_b_whole,crop_size, 0)

        # detect_results = None
        swap_result_list = []

        b_align_crop_tenor_list = []

        for b_align_crop in img_b_align_crop_list:

            b_align_crop_tenor = _totensor(cv2.cvtColor(b_align_crop,cv2.COLOR_BGR2RGB))[None,...].cuda()

            model_start = time.time()

            swap_result = model(None, b_align_crop_tenor, latend_id, None, True)[0]
            model_end = time.time()
            print ('model {}'.format(model_end-model_start))

            print ('prepare_input_MODEL {}'.format(model_start-prepare_input_MODEL))
            swap_result_list.append(swap_result)
            b_align_crop_tenor_list.append(b_align_crop_tenor)




        the_frame = img_b_whole       

        start_rev1 = time.time ()

        image_2 = reverse2wholeimage(b_align_crop_tenor_list, swap_result_list, b_mat_list, crop_size, the_frame, logoclass, \
            os.path.join(opt.output_path, 'result_whole_swapsingle.jpg'), opt.no_simswaplogo,pasring_model =net,use_mask=opt.use_mask, norm = spNorm)
        
        end_phase1 = time.time()
        print ('phase 1 :{}'.format (end_phase1-start_simswap))
        end_rev1 = time.time()
        print ('DONE rev1')
        rev1_time = end_rev1 - start_rev1
        print ('rev1_time : {}'.format(rev1_time))


        # img_c_align_crop_list, c_mat_list = detect_results
  

        # start_rev2 = time.time ()
        # # detect_results = None
        # swap_result_list = []

        # c_align_crop_tenor_list = []

        # for c_align_crop in img_c_align_crop_list:

        #     c_align_crop_tenor = _totensor(cv2.cvtColor(c_align_crop,cv2.COLOR_BGR2RGB))[None,...].cuda()

        #     # swap_result = model(None, b_align_crop_tenor, latend_id, None, True)[0]

        #     rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) 

        #     start = time.time()
        #     # Min-Max scaling
        #     min_val = np.min(rgb_image)
        #     max_val = np.max(rgb_image)
        #     norm_image = (rgb_image - min_val) / (max_val - min_val)

        #     # since output of simswap model standard is tensor (3,h,w) and normalized for input to rev2original and in BGR 
        #     swap_result = torch.from_numpy(norm_image.transpose(2,0,1))
        #     swap_result_list.append(swap_result)
        #     c_align_crop_tenor_list.append(c_align_crop_tenor)

        #     end = time.time()
        #     print ('pre ori 2 {}'.format(end-start))
        # image_2 = reverse2wholeimage([c_align_crop_tenor_list[min_index]], swap_result_list, [c_mat_list[min_index]], crop_size, whole_frame, logoclass, \
        #         os.path.join(opt.output_path, 'result_whole_swapsingle.jpg'), opt.no_simswaplogo,pasring_model =net,use_mask=opt.use_mask, norm = spNorm)

        # end_rev2 = time.time()
        # print ('DONE rev2')
        # rev2_time = end_rev2 - start_rev2
        # print ('rev2_time : {}'.format(rev2_time))

        end_simswap = time.time()
        print ('DONE SIMSWAP')
        simswap_time = end_simswap - start_simswap
        print ('SIMSWAP : {}'.format(simswap_time))

        # if opt.show_intermediate_image == 1:

        #     cv2.imshow('seg_swap_result', img_b_whole)
        #     cv2.waitKey(4000)
        #     cv2.imshow('crop_face_target', img_b_align_crop_list[0])
        #     cv2.waitKey(4000)

        return  image_2
 