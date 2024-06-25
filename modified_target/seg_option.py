from options.test_options import TestOptions
class SegOptions(TestOptions):
    def initialize (self):
        TestOptions.initialize(self)
        self.parser.add_argument("--config", default ='seg_swap/config/vox-256-sem-10segments.yaml', help="path to config")
        self.parser.add_argument("--checkpoint", default='seg_swap/checkpoint/vox-first-order.pth.tar', help="path to checkpoint to restore")
        self.parser.add_argument("--target_image", default='sup-mat/source.pn', help="path to target image to swap")
        self.parser.add_argument("--source_image", default='sup-mat/source.png', help="path to source image if not generate gan")
### SAVE GAN AND SAVE GRID
        self.parser.add_argument("--gan_face", type = int, default = 1, help="generate gan face")
        self.parser.add_argument("--show_grid", default = False, help="show_grid_swap_image")
        self.parser.add_argument("--save_gan_image_path", default='output/gan_image.jpg', help="path to the gan image path")
        self.parser.add_argument("--save_grid_path", default='output/grid.png', help="path to the grid image path")
### VIDEO 
        self.parser.add_argument("--the_video_path", default='sup-mat/source.png', help="path to the video path")
        self.parser.add_argument("--temp_results_dir", default='temp', help="path to temp_results_dir")
        self.parser.add_argument("--save_video_path", default='save_video/video.mp4', help="path to save_video_path, need .mp4 for no bug about codec")
### SEG SWAP 
        self.parser.add_argument("--swap_index", default="17,18", type=lambda x: list(map(int, x.split(','))),
                            help='index of swaped parts') ### 17, 18 means hair and hat 
        self.parser.add_argument("--hard", default = True , action="store_true", help="use hard segmentation labels for blending")
        self.parser.add_argument("--use_source_segmentation", action="store_true", help="use source segmentation for swaping")
        self.parser.add_argument("--first_order_motion_model", default = True , action="store_true", help="use first order model for alignment")
        self.parser.add_argument("--supervised", default = True , action="store_true",
                            help="use supervised segmentation labels for blending. Only for faces.")
        self.parser.add_argument("--cpu", action="store_true", help="cpu mode")
        self.parser.add_argument("--num_seg", type = int, default=3, help="number of segment times")

### OTHER SWAP OPTION
        self.parser.add_argument("--swap_option", type = str, default='swap_specific', help="swap_option")

        self.parser.add_argument("--use_mask_face_part_id_list", default='1,2,3,4,5,6,10,12,13',type=lambda x: list(map(int, x.split(','))),
                            help='use_mask_face_part_id_list')
        self.parser.add_argument("--visualize", default = False, help="frame_path")
        self.parser.add_argument("--bbox_modify", type = int, default=0, help="landmark coordinate change in x axix -> bbox_modify")
        self.parser.add_argument("--source_bbox_modify", type = int, default=0, help="landmark coordinate change in x axix -> bbox_modify")


        self.parser.add_argument("--det_thres_gan_or_source_image", type = float, default=0.3, help="det_thres for app.get")

        self.parser.add_argument("--det_thres_frame_for_segswap", type = float, default=0.2, help="det_thres for app.get")
        self.parser.add_argument("--det_thres_frame_for_simswap", type = float, default=0.2, help="det_thres for app.get for frame for simswap ")
        self.parser.add_argument("--take_pic_specific", type = int, default=0, help="open webcam to take pic_specific ")
        self.parser.add_argument("--simswap_crop_size", type = int, default=224, help="crop size simswap model")
        self.parser.add_argument("--seg_crop_size", type = int, default=256, help="crop size segment model")
        self.parser.add_argument("--return_grid", type = int, default=0, help="return grid in output folder")
        self.parser.add_argument("--collect_result_mode", type = int, default=0, help="collect_result_mode")
        self.parser.add_argument("--num_store", type = int, default=0, help="folder number store collect result")
        self.parser.add_argument("--more_hair_rev", type = int, default=0, help="more hair reverse back to image - trade off: border more visible")

        self.parser.add_argument("--user_choose_activate", type = int, default=0, help="activate user gan choose gender and age ")
        self.parser.add_argument("--user_choose", default='0',type=lambda x: list(map(int, x.split(','))),
                            help='1,20 meaning 1: gender male , 20 meaning age 20')



