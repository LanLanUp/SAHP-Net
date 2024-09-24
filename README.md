# Coarse for Fine: Bounding Box Supervised Thyroid Ultrasound Image Segmentation Using Spatial Arrangement and Hierarchical Prediction Consistency
A novel bounding box supervised segmentation framework is proposed to optimize the backbone network by measuring the rationality of the spatial distribution of high-level features, rather than the affinity of blurred, noisy low-level features, avoiding the problem of segmentation over-fitting bounding box annotations. **The spatial arrangement consistency(SAC) branch** is designed to compare only the horizontal and vertical maximum activations sampled from the preliminary segmentation prediction and the bounding box mask, facilitating the network to truly learn the differences between features under the loose constraints, rather than rigidly converging on the box labels. **The hierarchical prediction consistency(HPC) branch** is presented to compare the preliminary segmentation prediction and the secondary segmentation prediction, which is induced by the prototypes encapsulated from the semantic features weighted by the preliminary segmentation prediction and the bounding box mask together, guiding the network to distinguish target and background features with rational, elaborate spatial distribution.
![The overall architecture of our proposed framework.](https://github.com/user-attachments/assets/48c60b95-041b-4d01-b181-1feac7451520)
## Installation
### Requirements:
* Python=3.8
* Pytorch=1.11.0
* Detectron2: follow Detectron2 installation instructions.
* pip install -r requirements.txt
### Deformable Attention Block(Option)
```
cd ./modeling/layers/deform_attn
sh ./make.sh
cd ./modeling/layers/diff_ras
python setup.py build install
```
### Conda Environment Setup
```
conda create --name sahanet python=3.8 -y
conda activate boxsnake

conda install pytorch==1.9.0 torchvision==0.10.0 cudatoolkit=11.1 -c pytorch -c nvidia
python -m pip install 'git+https://github.com/facebookresearch/detectron2.git'

git clone https://github.com/LinG24/SaHaNet.git
cd SaHaNet
pip install -r requirements.txt
bash scripts/auto_build.sh
## Start
```
python train_net.py --config-file configs\Thyroid-Segmentation\saha_sampling.json
```
