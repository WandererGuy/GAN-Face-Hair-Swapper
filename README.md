## Official code for paper "Privacy-preserving Face and Hair Swapping in Real-time with a GAN-generated Face Image", publication in IEEE Access (2024)
[![DOI](https://img.shields.io/badge/Paper-PDF-red.svg)]([https://arxiv.org/abs/2108.08186](https://ieeexplore.ieee.org/document/10577121))

Welcome to our code base ^v^ We glad you are here. We hope this code is useful for you and the research community.

please use this code responsibly and legally , we strongly prohibit unethical action that involves swap other's face for personal gain without their consent<br>

First: git clone this repo.

## Preparation 
Brief review <br>
Checkpoint arcface, simswap same as checkpoint on https://github.com/neuralchen/SimSwap/blob/main/docs/guidance/preparation.md <br>
Checkpoint age and gender by us<br>
Checkpoint controllable GAN same as on https://github.com/amazon-science/gan-control <br>
Checkpoint for hair swap model same as on https://github.com/AliaksandrSiarohin/motion-cosegmentation?tab=readme-ov-file <br>

For convenience, here is all checkpoints in google drive https://drive.google.com/file/d/1TTH619l82Lc9uB3CYILFgkqMb9xoaQ-q/view?usp=sharing <br>
Step 1: Download and Unzip the file <br>
Step 2: go to ./move_checkpoint.py and change checkpoint_folder variable to the your unzip folder path <br>
Step 3: run <br>
```
python move_checkpoint.py  
```
(this will make directory and move checkpoints to its places)

## Explain what happen behind the scene of move_checkpoint.py (you can skip this part if u are not interested)
go into all_checkpoints folder, and you will see many folders , each have checkpoint(s) in it. <br>
Each folder has name , which is the path the script will create to put the corresponding checkpoint in that path .<br>
If the path already exist in your code, the script doesn't have to make the path, just put the checkpoint in existed path <br>

Example : <br>
Folder name: insightface_func+models+antelope  <br>
Meaning that: the main code, script will create a path: insightface_func/models/antelope <br>
and put the checkpoint belongs to insightface_func+models+antelope into insightface_func/models/antelope folder script just created.<br>
Fun fact: as you can see, the separator '+' in what separates directories 

## Environment 
This is what we install on our server<br>
Our server spec : 16.04 ubuntu , CUDA 11.3 (shown in nvidia-smi) <br>

```
conda create --name face_env python==3.9
conda activate face_env
conda install pytorch==1.12.1 torchvision==0.13.1 torchaudio==0.12.1 cudatoolkit=11.3 -c pytorch
python3 -m pip install onnxruntime-gpu
python3 -m pip install insightface==0.2.1 onnxruntime moviepy
conda install -c conda-forge pretrainedmodels
conda install -c anaconda pandas
conda install conda-forge::dlib
conda install anaconda::yaml
conda install anaconda::pyyaml
```

## Usage <br>
### 1. swap source image 's GAN into target image <br>
```
python 2_swap_image.py --name people --Arc_path arcface_model/arcface_checkpoint.tar --use_source_segmentation --source_image Downloads/di.jpg --pic_specific_path Downloads/bean.jpg --target_image Downloads/bean.jpg --num_seg 1 --swap_index 17,18 --show_grid True --gan_face 1  --bbox_modify 30 --use_mask 
```
expected output: results will be saved in ./ALL_TEST_IMAGE
### 2.  swap video , save frame into ./temp folder
```
python 1_process_video.py --name people --Arc_path arcface_model/arcface_checkpoint.tar --use_source_segmentation --source_image Downloads/di.jpg --pic_specific_path Downloads/bean.jpg --num_seg 1 --swap_index 17,18 --show_grid True --gan_face 1  --bbox_modify 30 --use_mask --the_video_path <your_video_path_here>
```
create video from ./temp folder and save video in ./save_video folder
```
python 4_make_video_temp.py 
```
### 3. swap webcam 
```
python 3_webcam_swap.py --name people --Arc_path arcface_model/arcface_checkpoint.tar --use_source_segmentation --source_image Downloads/di.jpg --pic_specific_path Downloads/bean.jpg --num_seg 1 --swap_index 17,18 --show_grid True --gan_face 1  --bbox_modify 30 --use_mask 
```
## parameter explain
| Parameter                | Description                                                                                                                                                                                                                       |
|--------------------------|-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| `--name`                 | SimSwap checkpoint name                                                                                                                                                                                                          |
| `--Arc_path`             | Arcface model path                                                                                                                                                                                                               |
| `--use_source_segmentation` | Activate hair swap, prefer source swap                                                                                                                                                                                           |
| `--pic_specific_path`    | Specific person path to swap (can also use target person image path; useful if there are many people in a picture as Arcface can compare face IDs and swap only the target person); use the same value as `--target_image` if only one person is in the target image |
| `--num_seg`              | Number of times to swap hair (best quality, slow: 3; fair quality, fast: 1)                                                                                                                                                        |
| `--swap_index`           | 17, 18 is for swapping hair                                                                                                                                                                                                       |
| `--target_image`         | Target image path (target image to swap into)                                                                                                                                                                                     |
| `--show_grid`            | Save a grid of processed images in a folder (specified in running script)                                                                                                                                                         |
| `--gan_face`             | 0: use source to swap, no need to generate GAN face; 1: use GAN to swap and source used to generate GAN based on age and gender                                                                                                    |
| `--source_image`         | Source person image path                                                                                                                                                                                                         |
| `--bbox_modify`          | Best is 30; the crop of head crop                                                                                                                                                                                                 |
| `--use_mask`             | For smoothening and blurring border once swapped image is put back into original image, but takes longer time                                                                                                                     |

## possible furture improvement 
- fix the file path for easier than read name path
- cooperate idea for faster inference with MobileSwap for Simswap + hairstyleGAN (possible face tracking for faster reID, skip frame, smaller image size)
- make docker version of this code base and above 
- make a colab version



## Few notes:
some parts of code is our server path , so if you encounter those as bug , just simply replace with your path. 
Any question , feel free to open up issue <3 We welcome feedbacks :>

## Troubleshooting

The best way to find and solve your problems is to see in the github issue tab. If you can't find what you want, feel free to raise an issue.

## Contributing

Contributions are what make the open source community such an amazing place to be learn, inspire, and create. Any contributions you make are **greatly appreciated**.

1. Fork the Project
1. Create your Feature Branch (`git checkout -b feature/AmazingFeature`)
1. Run `make style && make quality` in the root repo directory, to ensure code quality.
1. Commit your Changes (`git commit -m 'Add some AmazingFeature'`)
1. Push to the Branch (`git push origin feature/AmazingFeature`)
1. Open a Pull Request

## [Cite our paper](https://arxiv.org/abs/2108.08186)

```bibtex
@misc{
}
```
