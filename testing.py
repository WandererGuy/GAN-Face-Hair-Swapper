from __future__ import division

# from insightface_func.face_detect_crop_multi import Face_detect_crop
import cv2
import collections
import numpy as np
import glob
import os
import os.path as osp
import cv2
from insightface.model_zoo import model_zoo
from insightface_func.utils import face_align_ffhqandnewarc as face_align

__all__ = ['Face_detect_crop', 'Face']

Face = collections.namedtuple('Face', [
    'bbox', 'kps', 'det_score', 'embedding', 'gender', 'age',
    'embedding_norm', 'normed_embedding',
    'landmark'
])

Face.__new__.__defaults__ = (None, ) * len(Face._fields)


class Face_detect_crop:
    def __init__(self, name, root='~/.insightface_func/models'):
        self.models = {}
        root = os.path.expanduser(root)
        onnx_files = glob.glob(osp.join(root, name, '*.onnx'))
        onnx_files = sorted(onnx_files)
        for onnx_file in onnx_files:
            if onnx_file.find('_selfgen_')>0:
                #print('ignore:', onnx_file)
                continue
            model = model_zoo.get_model(onnx_file)
            if model.taskname not in self.models:
                print('find model:', onnx_file, model.taskname)
                self.models[model.taskname] = model
            else:
                print('duplicated model task type, ignore:', onnx_file, model.taskname)
                del model
        assert 'detection' in self.models
        self.det_model = self.models['detection']


    def prepare(self, ctx_id, det_thresh=0.5, det_size=(640, 640), mode ='None'):
        self.det_thresh = det_thresh
        self.mode = mode
        assert det_size is not None
        print('set det-size:', det_size)
        self.det_size = det_size
        for taskname, model in self.models.items():
            if taskname=='detection':
                model.prepare(ctx_id, input_size=det_size)
            else:
                model.prepare(ctx_id)

    def get(self, img, crop_size, bbox_modify, max_num=0):
        bboxes, kpss = self.det_model.detect(img,
                                             threshold=self.det_thresh,
                                             max_num=max_num,
                                             metric='default')
        print ('#####')
        print (bboxes.shape)
        print (kpss.shape)
        if bboxes.shape[0] == 0:
            return None
 
        align_img_list = []
        M_list = []



        # print ('BBBOXXXXX')
        # print (bboxes.shape[0])
        for i in range(bboxes.shape[0]):
            kps = None
            if kpss is not None:
                kps = kpss[i]

                # print (type(kps))
                # print (kps)

                # changing landmark coordinate in eye , lip leads to different affine matrix M -> bigger bounding box by test 
                if i == 0:
                    print (bboxes)
                    print (bboxes.shape[0])
                    print ('#######')
                    print (kps[0,0])
                    print (kps[0,1])
                    print (kps[1,0])
                    print (kps[1,1])
                    print (kps[3,0])
                    print (kps[3,1])
                    print (kps[4,0])
                    print (kps[4,1])
                    print ('#######')

                    kps[0,0] -= bbox_modify/1.5
                    kps[0,1] -= bbox_modify
                    kps[1,0] += bbox_modify/1.5
                    kps[1,1] -= bbox_modify
                    kps[3,0] -= bbox_modify/1.5
                    kps[3,1] += bbox_modify 
                    kps[4,0] += bbox_modify/1.5
                    kps[4,1] += bbox_modify


            M, _ = face_align.estimate_norm(kps, crop_size, mode = self.mode) 
            align_img = cv2.warpAffine(img, M, (crop_size, crop_size), borderValue=0.0)
            align_img_list.append(align_img)
            M_list.append(M)
            print ('%%%%%%%%')
            print (M)
        return align_img_list, M_list


if __name__ == '__main__':
    mode = 'None'
    app = Face_detect_crop(name='antelope', root='./insightface_func/models')
    app.prepare(ctx_id= 0, det_thresh=0.05, det_size=(640,640),mode=mode)

    img = cv2.imread('/home/manh/Downloads/musk.jpg')
############### get image B (crop face)##############

    align_img, M = app.get(img,256, 20)
################# try get cooordate in A (where pic B cut from)  ########
    ####### method A -> affine M -> got B 
    ####### so got B try get coordinate in A somehow 
    imgB = align_img[0]
    # Get crop size
    h, w = imgB.shape[:2]

    # Define crop corners in B
    pts_B = np.float32([[0,0], [w,0], [w, h], [0, h]]) 

    print ('GAYYY')
    print (pts_B)
    # Invert affine transform
    Minv = cv2.invertAffineTransform(M[0])

    # Transform crop corners back to image A 
    pts_A = cv2.perspectiveTransform(np.array([pts_B]), Minv)

    # Print transformed points
    # print(pts_A.reshape(-1, 2))


    # cv2.waitKey(5000)

    # Load image 

    ############## once obtain coordinate in A -> rectangle in pic######
    img = img
    coords = np.array(pts_A.reshape(-1, 4))

    # Draw bounding box 
    print (coords)
    cv2.rectangle(img, (int(coords[0][0]), int(coords[0][1])), 
                (int(coords[0][2]), int(coords[0][3])), (0,255,0), 2) 

    # Show image with bounding box
    cv2.imshow('Image A', img)
    cv2.waitKey(5000)