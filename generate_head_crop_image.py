from parsing_model.model import BiSeNet
from util.norm import SpecificNorm
import torch
import cv2
import os
import numpy as np
from skimage.measure import label, regionprops, find_contours

def _totensor(array):
    """Convert a NumPy array to a PyTorch tensor."""
    tensor = torch.from_numpy(array)
    return tensor.permute(2, 0, 1).float() / 255

# Load the model outside of the function
n_classes = 19
net = BiSeNet(n_classes=n_classes)
net.cuda()
save_pth = os.path.join('./parsing_model/checkpoint', '79999_iter.pth')
net.load_state_dict(torch.load(save_pth))
net.eval()

def create_segment_mask(pic):
    """Create a segmentation mask for the input image."""
    spNorm = SpecificNorm()
    pic_tensor = _totensor(cv2.cvtColor(pic, cv2.COLOR_BGR2RGB))[None, ...].cuda()
    source_img_norm = spNorm(pic_tensor)
    out = net(source_img_norm)[0]
    return out

def make_mask(pic):
    segment_mask_array = create_segment_mask(pic)
    squeezed_array = torch.squeeze(segment_mask_array, dim=0)
    argmax_mask = squeezed_array.argmax(dim=0)
    mask = ((argmax_mask >= 1) & (argmax_mask <= 13)) | (argmax_mask == 17)
    numpy_mask = mask.cpu().numpy().astype(np.uint8) * 255
    return numpy_mask

def mask_to_border(mask):
    h, w = mask.shape
    border = np.zeros((h, w))
    contours = find_contours(mask, 128)
    for contour in contours:
        for c in contour:
            x = int(c[0])
            y = int(c[1])
            border[x][y] = 255
    return border

def mask_to_bbox(mask):
    bboxes = []
    mask = mask_to_border(mask)
    lbl = label(mask)
    props = regionprops(lbl)
    for prop in props:
        x1, y1, x2, y2 = prop.bbox
        bboxes.append([x1, y1, x2, y2])
    return bboxes

def cropped_image(image):
    mask = make_mask(image)
    bboxes = mask_to_bbox(mask)
    areas = [(box[2] - box[0]) * (box[3] - box[1]) for box in bboxes]
    largest_index = np.argmax(areas)
    largest_bbox = bboxes[largest_index]

    x_min, y_min, x_max, y_max = largest_bbox
    center_x = (x_min + x_max) // 2
    center_y = (y_min + y_max) // 2
    side_length = max(x_max - x_min, y_max - y_min)
    new_x_min = center_x - (side_length // 2)
    new_x_max = center_x + (side_length // 2)
    new_y_min = center_y - (side_length // 2)
    new_y_max = center_y + (side_length // 2)
    cropped = image[new_y_min:new_y_max, new_x_min:new_x_max, :]
    return cropped, new_x_min, new_x_max, new_y_min, new_y_max

def generate_head_crop(image):
    crop_head, new_x_min, new_x_max, new_y_min, new_y_max = cropped_image(image)
    return crop_head, new_x_min, new_x_max, new_y_min, new_y_max



'''
from parsing_model.model import BiSeNet
from util.norm import SpecificNorm
import torch
import cv2
import os
import numpy as np
import time 
from skimage.measure import label, regionprops, find_contours

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
    segment_mask_array = create_segment_mask(pic)
    end1 = time.time()
    squeezed_array = np.squeeze(segment_mask_array, axis=0)
    new_arr = squeezed_array.permute(1, 2, 0)
    arr = np.zeros((h, w, 1), dtype=np.uint8)
    mask = (new_arr.argmax(axis=2) >= 1) & (new_arr.argmax(axis=2) <= 13) | (new_arr.argmax(axis=2) == 17)
    # mask = (new_arr.argmax(axis=2) == 17)

    numpy_mask = mask.detach().cpu().numpy()
    arr[numpy_mask] = 255  # Set regions of interest to white (255)
    # Convert the binary mask to grayscale
    gray_mask = arr[:, :, 0]


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
    areas = [(box[2] - box[0]) * (box[3] - box[1]) for box in bboxes]
    largest_index = torch.argmax(torch.tensor(areas))
    largest_bbox = bboxes[largest_index]

    x_min, y_min, x_max, y_max = largest_bbox
    center_x = (x_min + x_max) // 2
    center_y = (y_min + y_max) // 2
    side_length = max(x_max - x_min, y_max - y_min)
    new_x_min = center_x - (side_length // 2)
    new_x_max = center_x + (side_length // 2)
    new_y_min = center_y - (side_length // 2)
    new_y_max = center_y + (side_length // 2)
    image = image[new_y_min:new_y_max, new_x_min:new_x_max, :]
    return image, new_x_min, new_x_max, new_y_min, new_y_max 

def generate_head_crop(image):

    # image = cv2.imread(, cv2.IMREAD_COLOR) #Ảnh đầu vào phải được đọc màu (Với ảnh KhÁ bẢnH sau khi đọc xong là (400, 600, 3))
    crop_head, new_x_min, new_x_max, new_y_min, new_y_max = cropped_image(image) #Đầu ra của hàm này là ảnh đã crop theo bbox rồi. Những dòng sau chỉ để visualize


    return crop_head, new_x_min, new_x_max, new_y_min, new_y_max
# if __name__ == "__main__":
#     image = cv2.imread("/home/phungductung/Downloads/leonardo.jpg", cv2.IMREAD_COLOR) #Ảnh đầu vào phải được đọc màu (Với ảnh KhÁ bẢnH sau khi đọc xong là (400, 600, 3))
#     im = cropped_image(image) #Đầu ra của hàm này là ảnh đã crop theo bbox rồi. Những dòng sau chỉ để visualize
#     cv2.imshow("cropped", im)
#     while True:
#         key = cv2.waitKey(1)  
#         if key == ord('q'):
#             break
#     cv2.destroyAllWindows()
        

'''