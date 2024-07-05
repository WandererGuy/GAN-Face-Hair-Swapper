import glob
import os 
from moviepy.video.io.ImageSequenceClip import ImageSequenceClip
from modified_target.seg_option import SegOptions


def make_video_temp(opt):  
    os.makedirs('save_video', exist_ok=True)
    temp_results_dir = opt.temp_results_dir
    path = os.path.join(temp_results_dir,'*.jpg')
    image_filenames = sorted(glob.glob(path))
    clips = ImageSequenceClip(image_filenames,fps = 30)
    save_video_path = opt.save_video_path
    # clips.write_videofile(save_video_path,audio_codec='aac')
    # if not no_audio:
    #     clips = clips.set_audio(video_audio_clip)
    clips.write_videofile(save_video_path,audio_codec='aac')


if __name__ == '__main__':
    opt = SegOptions().parse()    
    make_video_temp(opt)

