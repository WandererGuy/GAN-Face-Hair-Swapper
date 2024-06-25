## official code for paper "Privacy-preserving Face and Hair Swapping in Real-time with a GAN-generated Face Image", publication in IEEE Access (2024)
## preparation 
checkpoint arcface, simswap same as checkpoint on https://github.com/neuralchen/SimSwap/blob/main/docs/guidance/preparation.md <br>
checkpoint age and gender ... <br>
checkpoint controllable GAN same as on https://github.com/amazon-science/gan-control <br>
checkpoint for hair swap model same as on https://github.com/AliaksandrSiarohin/motion-cosegmentation?tab=readme-ov-file <br>

## environment <br>
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
