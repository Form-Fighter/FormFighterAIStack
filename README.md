# FormFighterAIStack
Proof of Concept AI application that analyzes the fighting form of a person from a video using computer vision and provides natural language feedback.


# Download models
Models required to run WHAM can be found here: https://drive.google.com/drive/folders/1XNLxVJi4csWatAUsdDnF22jEbjluSJvA?usp=sharing

The file structure where they should be placed are:

/WHAM/checkpoints:
dpvo.pth
hmr2a.ckpt
vitpose-h-multi-coco.pth
wham_vit_bedlam_w_3dpw.pth.tar
wham_vit_w_3dpw.pth.tar
yolov8x.pt

/WHAM/dataset/body_models
J_regressor_coco.npy
J_regressor_feet.npy
J_regressor_h36m.npy
J_regressor_wham.npy
smpl_faces.npy
smpl_faces_f.npy
smpl_mean_params.npz
smplx2smpl.pkl

/WHAM/dataset/body_models/smpl
SMPL_FEMALE.pkl
SMPL_MALE.pkl
SMPL_NEUTRAL.pkl

# Installation
Installation instructions were modified by this guide: [click](https://github.com/yohanshin/WHAM/blob/main/docs/INSTALL.md)

Modified instructions:
![modified_install_instructions](https://github.com/user-attachments/assets/56a2568a-b0ab-4cf0-9115-dd0a5c33c05a)

## Clone the repo
git clone https://github.com/yohanshin/WHAM.git --recursive
cd WHAM/

## Create Conda environment
conda create -n wham python=3.9
conda activate wham

## Install PyTorch libraries
conda install pytorch==2.5.0 torchvision==0.20.0 torchaudio==2.5.0 cudatoolkit=12.4 -c pytorch

## Install dependencies for PyTorch3D (optional) for visualization
conda install -c fvcore -c iopath -c conda-forge fvcore iopath

## Install WHAM and ViTPose dependencies
pip install -r requirements.txt

## Install DPVO
cd third-party/DPVO
conda install pytorch-scatter=2.0.9 -c rusty1s
conda install cudatoolkit-dev=11.3.1 -c conda-forge

## ONLY IF your GCC version is larger than 10
conda install -c conda-forge gxx=9.5

pip install .


