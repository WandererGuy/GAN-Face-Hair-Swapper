## official code for paper "Privacy-preserving Face and Hair Swapping in Real-time with a GAN-generated Face Image", publication in IEEE Access (2024)
Welcome to our code base ^v^ We glad you are here. We hope this code is useful for you and the research community.
## preparation 
brief review <br>
checkpoint arcface, simswap same as checkpoint on https://github.com/neuralchen/SimSwap/blob/main/docs/guidance/preparation.md <br>
checkpoint age and gender by us<br>
checkpoint controllable GAN same as on https://github.com/amazon-science/gan-control <br>
checkpoint for hair swap model same as on https://github.com/AliaksandrSiarohin/motion-cosegmentation?tab=readme-ov-file <br>

For convenience, here is all checkpoints in google drive<br>
Step 1: Download and Unzip the file <br>
Step 2: go into all_checkpoints folder, and you will see many folders , each have checkpoint(s) in it. <br>
Step 3: Each folder has name , which is the path you must create to put the corresponding checkpoint in that path <br>
if the path already exist in your code, you don't have to make the path, just put the checkpoint in existed path <br>

Example: <br>
folder name: insightface_func+models+antelope  <br>
meaning that: the main code, you must create a path: insightface_func/models/antelope <br>
and put the checkpoint belongs to insightface_func+models+antelope into insightface_func/models/antelope folder you just created.<br>
Fun fact: as you can see, the separator '+' in the name separates directories 

## environment 
our spec : 16.04 ubuntu , CUDA 11.3 (in nvidia-smi) <br>

conda create -p /home/ai-ubuntu/hddnew/Manh/face_anon/face_9/face/face_env python==3.9 <br>
conda activate /home/ai-ubuntu/hddnew/Manh/face_anon/face_9/face/face_env <br>
conda install pytorch==1.12.1 torchvision==0.13.1 torchaudio==0.12.1 cudatoolkit=11.3 -c pytorch <br>
python3 -m pip install onnxruntime-gpu <br>
python3 -m pip install insightface==0.2.1 onnxruntime moviepy <br>
conda install -c conda-forge pretrainedmodels <br>
conda install -c anaconda pandas <br>
conda install conda-forge::dlib <br>
conda install anaconda::yaml <br>
conda install anaconda::pyyaml <br>

## usage <br>
python 2_swap_image.py --name people --Arc_path arcface_model/arcface_checkpoint.tar --use_source_segmentation --pic_specific_path Downloads/mr_bean.jpeg --num_seg 1 --swap_index 17,18 --target_image Downloads/mr_bean.jpeg --show_grid True --gan_face 0 --source_image Downloads/di.jpg --bbox_modify 30 --use_mask <br>
