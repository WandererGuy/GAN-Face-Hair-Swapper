import torch
import torch.nn.functional as F

import matplotlib.patches as mpatches
from modified_target.seg_option import SegOptions
import matplotlib.pyplot as plt
import numpy as np 

import cv2





def sub_visualize_segmentation(image, network, supervised=False, hard=True, colormap='gist_rainbow'):
    with torch.no_grad():
        inp = torch.tensor(image[np.newaxis].astype(np.float32)).permute(0, 3, 1, 2).cuda()
        if supervised:
            inp = F.interpolate(inp, size=(512, 512))
            inp = (inp - network.mean) / network.std
            mask = torch.softmax(network(inp)[0], dim=1)
            mask = F.interpolate(mask, size=image.shape[:2])
        else:
            mask = network(inp)['segmentation']
            mask = F.interpolate(mask, size=image.shape[:2], mode='bilinear')

    if hard:
        mask = (torch.max(mask, dim=1, keepdim=True)[0] == mask).float()

    colormap = plt.get_cmap(colormap)
    num_segments = mask.shape[1]
    mask = mask.squeeze(0).permute(1, 2, 0).cpu().numpy()
    color_mask = 0
    patches = []
    for i in range(num_segments):
        if i != 0:
            color = np.array(colormap((i - 1) / (num_segments - 1)))[:3]
        else:
            color = np.array((0, 0, 0))
        patches.append(mpatches.Patch(color=color, label=str(i)))
        color_mask += mask[..., i:(i+1)] * color.reshape(1, 1, 3)

    fig, ax = plt.subplots(1, 2, figsize=(12,6))

    ax[0].imshow(color_mask)
    ax[1].imshow(0.3 * image + 0.7 * color_mask)
    ax[1].legend(handles=patches)
    ax[0].axis('off')
    ax[1].axis('off')
    return fig,ax

# visualize_segmentation(source_image, segmentation_module, hard=True)
# plt.show()

def visualize_segmentation(source_image, target_image):
    from modified_target.seg_swap_model import load_face_parser
    import imageio
    from skimage.transform import resize
    import matplotlib

    matplotlib.use('TkAgg')


    face_parser = load_face_parser(cpu=False)

    # source_image = imageio.imread(source_image_path)
    source_image = resize(source_image, (256, 256))[..., :3]



    fig, ax = sub_visualize_segmentation(source_image, face_parser, supervised=True, hard=True, colormap='tab20')
    plt.show()
    # print ('YOOOOOOOOOOOOOOOOOOO')
    cv2.waitKey(3000)
    fig.savefig('/home/manh/coding/faceswap/output/source_semantic.png')
    plt.close()

    target_image = resize(target_image, (256, 256))[..., :3]
    fig, ax = sub_visualize_segmentation(target_image, face_parser, supervised=True, hard=True, colormap='tab20')
    plt.show()
    # print ('YOOOOOOOOOOOOOOOOOOO')
    cv2.waitKey(3000)
    fig.savefig('/home/manh/coding/faceswap/output/target_semantic.png')

    plt.close()