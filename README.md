# MCANet
This repository contains the implementation of MCANet: A Multi-Scale and Class-Specific Attention Network for Rapid Multi-Label Post-Hurricane Damage Assessment with UAV Images


## 📁 Dataset Preparation

1. **Unzip the provided data folder**  

2. **Download the raw UAV imagery**  
   The original RescueNet dataset can be downloaded from the following Dropbox link:

   🔗 [Download RescueNet UAV Images](https://www.dropbox.com/scl/fo/ntgeyhxe2mzd2wuh7he7x/AFIchlfjVO_7MzPcNc1ZOHE/RescueNet?rlkey=6vxiaqve2mzd2wuh7he7x&subfolder_nav_tracking=1&st=jtkzy0ob&dl=0)

3. **Generate multi-label annotations**  
   Run the following script to generate multi-label JSON annotations from the raw images:

   ```bash
   python utils/prepare/prepare_multilabel_rescuenet.py

This will create the required .json annotation files in the data/rescuenet/ directory.
 Note: Preprocessed label files have already been included in this repository under the data/ folder.
   
##🚀 Training & Evaluation
python main.py --model vit_B16_448 --num_heads 1 --img_size 448 --dataset rescuenet --num_cls 10

   
