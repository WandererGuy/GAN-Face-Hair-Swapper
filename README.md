## Official code for paper "Privacy-preserving Face and Hair Swapping in Real-time with a GAN-generated Face Image", publication in IEEE Access (2024)
paper link [https://ieeexplore.ieee.org/document/10577121](url)
Welcome to our code base ^v^ We glad you are here. We hope this code is useful for you and the research community.

First: git clone this repo.

## Preparation 
Brief review <br>
Checkpoint arcface, simswap same as checkpoint on https://github.com/neuralchen/SimSwap/blob/main/docs/guidance/preparation.md <br>
Checkpoint age and gender by us<br>
Checkpoint controllable GAN same as on https://github.com/amazon-science/gan-control <br>
Checkpoint for hair swap model same as on https://github.com/AliaksandrSiarohin/motion-cosegmentation?tab=readme-ov-file <br>

For convenience, here is all checkpoints in google drive https://drive.google.com/file/d/1TTH619l82Lc9uB3CYILFgkqMb9xoaQ-q/view?usp=sharing <br>
Step 1: Download and Unzip the file <br>
Step 2: go to ./move_checkpoint.py and change checkpoint_folder to the your unzip folder path
Step 3: run python move_checkpoint.py (this will make directory and move checkpoints to its places)

## Explain what happen behind the scene (you can skip this part if u are not interested)
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

conda create --name face_env python==3.9 <br>
conda activate face_env <br>
conda install pytorch==1.12.1 torchvision==0.13.1 torchaudio==0.12.1 cudatoolkit=11.3 -c pytorch <br>
python3 -m pip install onnxruntime-gpu <br>
python3 -m pip install insightface==0.2.1 onnxruntime moviepy <br>
conda install -c conda-forge pretrainedmodels <br>
conda install -c anaconda pandas <br>
conda install conda-forge::dlib <br>
conda install anaconda::yaml <br>
conda install anaconda::pyyaml <br>

## Usage <br>
python 2_swap_image.py --name people --Arc_path arcface_model/arcface_checkpoint.tar --use_source_segmentation --pic_specific_path Downloads/mr_bean.jpeg --num_seg 1 --swap_index 17,18 --target_image Downloads/mr_bean.jpeg --show_grid True --gan_face 0 --source_image Downloads/di.jpg --bbox_modify 30 --use_mask <br>

expected output: results will be saved in ./ALL_TEST_IMAGE

## possible furture improvement 
- fix the file path for easier than read name path
- cooperate idea for faster inference with MobileSwap for Simswap + hairstyleGAN (possible face tracking for faster reID, skip frame, smaller image size)
- make docker version of this code base and above 
- make a colab version



## Few notes:
some parts of code is our server path , so if you encounter those as bug , just simply replace with your path. 
Any question , feel free to open up issue <3 We welcome feedbacks :>
