from parsing_model.model import BiSeNet
from util.norm import SpecificNorm
import torch
from modified_target.seg_option import SegOptions
import cv2
import torch.nn.functional as F
import os
import numpy as np
import time 

def _totensor(array):
    tensor = torch.from_numpy(array)
    img = tensor.transpose(0, 1).transpose(0, 2).contiguous()
    return img.float().div(255)


def create_segment_mask(pic):
        n_classes = 19
        net = BiSeNet(n_classes=n_classes)
        net.cuda()
        save_pth = os.path.join('./parsing_model/checkpoint', '79999_iter.pth')
        net.load_state_dict(torch.load(save_pth))
        net.eval()

        parsing_model = net
        spNorm =SpecificNorm()
        norm = spNorm

        pic_tenor = _totensor(cv2.cvtColor(pic,cv2.COLOR_BGR2RGB))[None,...].cuda()

        source_img = pic_tenor 

        source_img_norm = norm(source_img)
        source_img_512  = F.interpolate(source_img_norm,size=(512,512))
        out = parsing_model(source_img_512)[0]
        return out

if __name__ == '__main__':
        opt = SegOptions().parse()    
        pic = cv2.imread (opt.pic_specific_path)
        start = time.time()

        segment_mask_array = create_segment_mask(pic)
        # print (type(segment_mask_array))
        # print (segment_mask_array)
        # print (segment_mask_array.shape)
        print ('***********')

        ### go through each pixel and collect class list info
        # for segment_mask 
        # element = segment_mask_array [0,:,0,0]
        # print (element)

        ### select the class responsible for a pixel 
        ### a pixel have 19 probability for each class 
        ### collect correct class index for indexing 
        # max_index = torch.argmax(element)
        end1 = time.time()

        squeezed_array = np.squeeze(segment_mask_array, axis=0)
 
        new_arr = squeezed_array.permute(1,2,0)

        arr = np.zeros((512, 512, 1))

        # for i in range(new_arr.shape[0]):
        #         for j in range(new_arr.shape[1]):
        #                 # for k in range(new_arr.shape[2]):
        #                 array_class_value = new_arr[i, j, :]   
        #                 max_index = torch.argmax(array_class_value)
        #                 if 1<= max_index <=13 or max_index == 17:
        #                         arr [i,j,0] = 1

        mask = (new_arr.argmax(axis=2) >= 1) & (new_arr.argmax(axis=2) <= 13) | (new_arr.argmax(axis=2) == 17)
        # mask = (new_arr.argmax(axis=2) == 17) | (new_arr.argmax(axis=2) == 14) | (new_arr.argmax(axis=2) == 16) 
        # mask = (new_arr.argmax(axis=2) == 17) 




        numpy_mask = mask.detach().cpu().numpy()
        ### modify arr with arr[i,j,0] = 1 (the head become 1)
        arr[numpy_mask, 0] = 1

        end = time.time()
        print ('bisenet :{}'.format(end1-start))
        print ('all :{}'.format(end-start))
        cv2.imshow ('binary pic', arr)
        cv2.waitKey(5000)



