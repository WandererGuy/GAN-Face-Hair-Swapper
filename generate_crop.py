from parsing_model.model import BiSeNet
from util.norm import SpecificNorm
import torch
import cv2
import os
import numpy as np
import time 
from skimage.measure import label, regionprops, find_contours
from modified_target.seg_option import SegOptions


def _totensor(array): #OK
    tensor = torch.from_numpy(array)
    img = tensor.transpose(0, 1).transpose(0, 2).contiguous()
    return img.float().div(255)

def create_segment_mask(pic): #OK
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
        out = parsing_model(source_img_norm)[0]

        return out

def make_mask(pic):
    h = pic.shape[0]
    w = pic.shape[1]

    start = time.time()
    segment_mask_array = create_segment_mask(pic)
    print('*********')
    end1 = time.time()
    squeezed_array = np.squeeze(segment_mask_array, axis=0)
    new_arr = squeezed_array.permute(1, 2, 0)
### create a new numpy black with size like original image 
    arr = np.zeros((h, w, 1), dtype=np.uint8)

    mask = (new_arr.argmax(axis=2) >= 1) & (new_arr.argmax(axis=2) <= 13) | (new_arr.argmax(axis=2) == 17)
    numpy_mask = mask.detach().cpu().numpy()
    arr[numpy_mask] = 255  # Set regions of interest to white (255)
    # Convert the binary mask to grayscale
    gray_mask = arr[:, :, 0]

    end = time.time()
    print('bisenet :{}'.format(end1 - start))
    print('all :{}'.format(end - start))
    print(gray_mask.shape)
    return gray_mask

def mask_to_border(mask):
    h = mask.shape[0]
    w = mask.shape[1]
    border = np.zeros((h, w))
    contours = find_contours(mask, 128)
    for contour in contours:
        for c in contour:
            x = int(c[0])
            y = int(c[1])
            border[x][y] = 255
    return border

""" Mask to bounding boxes """
def mask_to_bbox(mask):
    bboxes = []
    mask = mask_to_border(mask)
    lbl = label(mask)
    props = regionprops(lbl)
    for prop in props:
        x1 = prop.bbox[1]
        y1 = prop.bbox[0]
        x2 = prop.bbox[3]
        y2 = prop.bbox[2]
        bboxes.append([x1, y1, x2, y2])
    return bboxes

def cropped_image(image):
    mask = make_mask(image)
    bboxes = mask_to_bbox(mask)
    for bbox in bboxes: 
        #image = cv2.rectangle(image, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (255, 0, 0), 2)
        
        image = image[bbox[1]:bbox[3], bbox[0]:bbox[2], :]
    return image     

if __name__ == "__main__":
    opt = SegOptions().parse()    

    image = cv2.imread(opt.pic_specific_path, cv2.IMREAD_COLOR) #Ảnh đầu vào phải được đọc màu (Với ảnh KhÁ bẢnH sau khi đọc xong là (400, 600, 3))
    im = cropped_image(image) #Đầu ra của hàm này là ảnh đã crop theo bbox rồi. Những dòng sau chỉ để visualize
    cv2.imshow("cropped", im)
    cv2.waitKey(5000)
        
