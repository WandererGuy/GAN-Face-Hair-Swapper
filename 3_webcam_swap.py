from modified_target.seg_option import SegOptions
from webcam import swap_webcam

if __name__ == '__main__':
        opt = SegOptions().parse()    
        swap_webcam(opt)
