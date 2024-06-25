import cv2
import torch
import fractions
from PIL import Image
import torch.nn.functional as F
from torchvision import transforms
from models.models import create_model
from insightface_func.face_detect_crop_multi import Face_detect_crop

def lcm(a, b): return abs(a * b) / fractions.gcd(a, b) if a and b else 0

def preprocess_segment(opt, pic_specific, gan_img):
    transformer = transforms.Compose([
        transforms.ToTensor(),
        #transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    transformer_Arcface = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])


    # pic_specific = opt.pic_specific_path
    bbox_modify = opt.bbox_modify
    seg_crop_size = opt.seg_crop_size

    torch.nn.Module.dump_patches = True
    if opt.simswap_crop_size == 512:
        opt.which_epoch = 550000
        opt.name = '512'
        mode = 'ffhq'
    else:
        mode = 'None'
    model = create_model(opt)
    model.eval()



#### initiate bbox 
    app = Face_detect_crop(name='antelope', root='./insightface_func/models')
    app.prepare(ctx_id= 0, det_thresh=opt.det_thres_frame_for_segswap, det_size=(640,640),mode=mode)

    # app_simswap = Face_detect_crop(name='antelope', root='./insightface_func/models')
    # app_simswap.prepare(ctx_id= 0, det_thresh=0.6, det_size=(640,640),mode=mode)


    # pic_a will be the specific target person to be swapped
    with torch.no_grad():
        if opt.gan_face:
            img_a_whole = gan_img 

        else:
            img_a_whole = cv2.imread(opt.source_image)

        # # img_a_align_crop, _ = app.get(img_a_whole,crop_size, opt.source_bbox_modify)
        # img,_,_,_,_ = generate_head_crop(img_a_whole)
        # img_a_align_crop = [img]
        # # cv2.imshow('head_source_crop', img_a_align_crop[0])
        # # cv2.waitKey(4000)

        # img_a_align_crop_pil = Image.fromarray(cv2.cvtColor(img_a_align_crop[0],cv2.COLOR_BGR2RGB)) 
        # img_a = transformer_Arcface(img_a_align_crop_pil)
        # img_id = img_a.view(-1, img_a.shape[0], img_a.shape[1], img_a.shape[2])


        # # convert numpy to tensor
        # img_id = img_id.cuda()
        # # img_att = img_att.cuda()

        # #create latent id
        # img_id_downsample = F.interpolate(img_id, size=(112,112))
        # latend_id = model.netArc(img_id_downsample)
        # latend_id = F.normalize(latend_id, p=2, dim=1)

        # The specific person to be swapped
        specific_person_id_nonorm = None 

        
        # specific_person_whole = cv2.imread(opt.target_video)
        if opt.swap_option == 'swap_specific':

            specific_person_whole = pic_specific
            #specific_person_whole = cv2.imread(pic_specific)


    ##### create bbox for target reference image 
            specific_person_align_crop, _ = app.get(specific_person_whole,seg_crop_size, 0)
            specific_person_align_crop_pil = Image.fromarray(cv2.cvtColor(specific_person_align_crop[0],cv2.COLOR_BGR2RGB)) 
            specific_person = transformer_Arcface(specific_person_align_crop_pil)
            specific_person = specific_person.view(-1, specific_person.shape[0], specific_person.shape[1], specific_person.shape[2])
            specific_person = specific_person.cuda()
            specific_person_downsample = F.interpolate(specific_person, size=(112,112))
            specific_person_id_nonorm = model.netArc(specific_person_downsample)


    return img_a_whole ,specific_person_id_nonorm, opt.id_thres, \
            model, app,opt.output_path,opt.no_simswaplogo,opt.use_mask,seg_crop_size



    