import os
import time
import random
import argparse
import numpy as np

import torch
import torch.nn.functional as F
from torch.backends import cudnn

from util import util
from util.plot import plot_batch

from models.projected_model import fsModel
from data.data_loader_Swapping import GetLoader

from modified_target.seg_option import SegOptions

from eval_phase import two_phases
from models.models import create_model
import cv2
from torchvision import transforms
from insightface_func.face_detect_crop_single import Face_detect_crop
from PIL import Image
import torch.nn as nn


transformer_Arcface = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

if __name__ == "__main__":
    opt = SegOptions().parse()


    # model = fsModel()
    crop_size = opt.crop_size

    # model.initialize(opt)
    final_img = two_phases(opt) 
    torch.cuda.empty_cache()    

    if crop_size == 512:
        opt.which_epoch = 550000
        opt.name = '512'
        mode = 'ffhq'
    else:
        mode = 'None'  
    app = Face_detect_crop(name='antelope', root='./insightface_func/models')
    app.prepare(ctx_id= 0, det_thresh=0.05, det_size=(640,640),mode=mode)

 

    torch.nn.Module.dump_patches = True
    model = create_model(opt)

    img_a_whole = cv2.imread(opt.source_image)


    img_a_align_crop, _ = app.get(img_a_whole,crop_size)
    img_a_align_crop_pil = Image.fromarray(cv2.cvtColor(img_a_align_crop[0],cv2.COLOR_BGR2RGB)) 
    img_a = transformer_Arcface(img_a_align_crop_pil)
    img_id = img_a.view(-1, img_a.shape[0], img_a.shape[1], img_a.shape[2])




    # img_id = np.transpose(img_id,(2,0,1))

    # img_id = torch.from_numpy(img_id)
    img_id = img_id.cuda()
    img_id_112      = F.interpolate(img_id,size=(112,112), mode='bicubic')
    latent_id       = model.netArc(img_id_112)
    latent_id       = F.normalize(latent_id, p=2, dim=1)

    img_fake = final_img 

    img_b_align_crop, _ = app.get(img_fake,crop_size)
    img_b_align_crop_pil = Image.fromarray(cv2.cvtColor(img_b_align_crop[0],cv2.COLOR_BGR2RGB)) 
    img_b = transformer_Arcface(img_b_align_crop_pil)
    img_fake = img_b.view(-1, img_b.shape[0], img_b.shape[1], img_b.shape[2])

    img_fake = img_fake.cuda()



    # final_img = np.transpose(final_img,(2,0,1))
    # final_img = torch.from_numpy(final_img)

    img_fake_down   = F.interpolate(img_fake, size=(112,112), mode='bicubic')

    latent_fake     = model.netArc(img_fake_down)
    latent_fake     = F.normalize(latent_fake, p=2, dim=1)
    loss_G_ID       = (1 - model.cosin_metric(latent_fake, latent_id)).mean()
    print (type(loss_G_ID) )

    # loss_G_ID = loss_G_ID.cpu().detach().numpy
    print (type(loss_G_ID) )
    print (loss_G_ID)

    print ('G_ID: '+str(loss_G_ID.item()))



# make target image suitable for rec function  cv2 format of crop 
    img_target_whole = cv2.imread(opt.frame_path)

    img_c_align_crop, _ = app.get(img_target_whole,crop_size)
    img_target = img_c_align_crop[0]
    # img_c_align_crop_pil = Image.fromarray(cv2.cvtColor(img_c_align_crop[0],cv2.COLOR_BGR2RGB)) 
    # img_c = transformer_Arcface(img_c_align_crop_pil)
    # img_target = img_c.view(-1, img_c.shape[0], img_c.shape[1], img_c.shape[2])




    # img_target = torch.from_numpy(img_target).cuda()



    # make fake image be like in simswap format for generated fake image 

    img_b_align_crop_cv = img_b_align_crop[0]

    # img_b_align_crop_cv = np.array(img_b_align_crop_pil)

    # img_b_align_crop_cv = cv2.cvtColor(img_b_align_crop_cv, cv2.COLOR_RGB2BGR)
    # img_b_align_crop[0],
    # img_b_align_crop_pil_cv = img_b_align_crop_pil
    # img_b_align_crop_cv = img_b_align_crop_cv.cpu().detach().numpy()
    print (img_b_align_crop_cv.shape)



    # img_b_align_crop_cv = np.transpose(img_b_align_crop_cv,(2,0,1))
    # img_b_align_crop_cv = torch.from_numpy(img_b_align_crop_cv)
    # img_b_align_crop_cv.cuda()

    print (img_b_align_crop[0].shape)
    print (img_target.shape)
    print (type(img_b_align_crop[0]))
    print (type(img_target))



    # criterionRec = nn.L1Loss
    # loss_G_Rec  = criterionRec(img_b_align_crop[0].all(), img_target.all())

    cv2.imshow ('webcam', img_b_align_crop[0])
    cv2.waitKey(2000)
    cv2.imshow ('webcam', img_target)
    cv2.waitKey(2000)

    img_b_align_crop[0] = img_b_align_crop[0].astype(np.cfloat)
    img_target = img_target.astype(np.cfloat)


    img_b_align_crop[0] = torch.from_numpy(img_b_align_crop[0] )
    img_target = torch.from_numpy(img_target)

    criterionRec = nn.L1Loss()
    loss_G_Rec  = criterionRec(img_b_align_crop[0], img_target)
    # loss_G_Rec.backward()
    # loss_G_Rec = np.mean(np.abs(img_b_align_crop[0] - img_target))

    # loss_G_Rec = loss_G_Rec * 10

    print (type(loss_G_Rec) )
    print (loss_G_Rec)

    print ('loss_G_Rec: '+str(loss_G_Rec))
    # two_phases(opt) 