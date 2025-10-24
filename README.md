# SHeRLoc: Synchronized Heterogeneous Radar Place Recognition for Cross-Modal Localization
**[IEEE RA-L 2025]** This repository is the official repository for SHeRLoc [**[Paper]**](https://arxiv.org/pdf/2506.15175) [**[Project]**](https://sites.google.com/view/radar-sherloc).

[Hanjun Kim](https://hanjun815.github.io/), [Minwoo Jung](https://scholar.google.co.kr/citations?user=aKPTi7gAAAAJ&hl=ko), [Wooseong Yang](https://scholar.google.co.kr/citations?hl=ko&user=lh2KUKMAAAAJ), [Ayoung Kim](https://scholar.google.co.kr/citations?user=7yveufgAAAAJ&hl=ko)†

 <div align="center">
    
  ![overview](https://github.com/user-attachments/assets/a3b908f3-896a-4af5-8923-90e99fda0997)

 </div>

## News
- 2025/10/08 : SHeRLoc is accepted in IEEE RA-L 2025.
- 2025/10/10 : Our project page is available via [**[Project]**](https://sites.google.com/view/radar-sherloc).
- 2025/10/23 : First release of SHeRLoc code.

## 0. Overall Pipeline
  ![overview](https://github.com/user-attachments/assets/434cf54f-593d-4cc4-90be-4a8c56e78969)

## 1. Environment
Code was tested using Python 3.8 with PyTorch 1.9.1 and MinkowskiEngine 0.5.4 on Ubuntu 20.04 with CUDA 10.2. Note: CUDA 11.1 is not recommended as there are some issues with MinkowskiEngine 0.5.4 on CUDA 11.1.
The following Python packages are required:
- PyTorch (version 1.9.1)
- MinkowskiEngine (version 0.5.4)
- pytorch_metric_learning (version 1.0 or above)
- wandb
- matplotlib


## 2. How to install
**2-1. Clone the Repository**

Clone the SHeRLoc repository and enter to the project directory:
```
git clone https://github.com/hanjun815/SHeRLoc.git
```

**2-2. Build the Docker Image**

Build the Docker image using the provided Dockerfile and start a container with GPU support.
Build the image:
```
cd SHeRLoc/docker
docker build -t sherloc .
```
Run the container:
```
docker run -it --rm --gpus all --net host --shm-size=1g -v  $(pwd)/../..:/code -v /data3:/data3 sherloc
```
**Note**: Adjust the docker run command to suit your environment.

Modify the PYTHONPATH environment variable to include absolute path to the project root folder:
```
export PYTHONPATH=$PYTHONPATH:/code/SHeRLoc
```

**2-3. Download the Preprocessed HeRCULES Dataset**

[**[Download HeRCULES Dataset]**](https://drive.google.com/drive/folders/1lrJg7MMfBzEEEyTwor3j6JysxaGwnR5i)

Move the downloaded `tar.gz` to each sequence directory inside `SHeRLoc/datasets/HeRCULES`, then extract it.
```
# example
cd SHeRLoc/datasets/HeRCULES/Bridge/01
tar -zxvf Continental.tar.gz
tar -zxvf Navtech.tar.gz
tar -zxvf Navtech_576.tar.gz
tar -zxvf Aeva.tar.gz
```
## 3. How to Execute
**3-0 Preprocessing**

Preprocess 4D radar point clouds, spinning radar scans, and LiDAR point clouds. Pass the folder containing the raw data to `--input`, and the folder where preprocessed results should be saved to `--output`. Preprocess all data you plan to use for training or evaluation in advance.

- Calibration with Mountain 01 Sequence
```
cd SHeRLoc/datasets
python calibration_continental.py
python calibration_navtech.py
python calibration_matching.pys
```
The heterogeneous﻿ radar calibration constant C<sub>correct</sub> obtained through calibration is already reflected in the preprocessing code (C<sub>correct</sub> = 2.1025).

- 4D Radar Preprocessing
```
cd SHeRLoc/datasets
python preprocess_4D.py --input HeRCULES/Bridge/01/raw_continental --output HeRCULES/Bridge/01/Continental
```
- Spinning Radar Preprocessing
```
cd SHeRLoc/datasets
python preprocess_Spinning.py --input HeRCULES/Bridge/01/raw_navtech --output HeRCULES/Bridge/01/Navtech
```
- LiDAR Preprocessing
```
cd SHeRLoc/datasets
python preprocess_LiDAR.py --input HeRCULES/Bridge/01/raw_aeva --output HeRCULES/Bridge/01/Aeva
```
- Spinning Radar Preprocessing (384*576 size image with 360 degree FOV for homogeneous Spinning radar PR)
```
cd SHeRLoc/datasets
python preprocess_Spinning_576.py --input HeRCULES/Bridge/01/raw_navtech --output HeRCULES/Bridge/01/Navtech_576
```

**3-1 Training**

**Train with the standard model (SHeRLoc):**
- Heterogeneous Radar Place Recognition
```
cd SHeRLoc/training
python train.py --config ../config/config.txt --model_config ../models/Hetero_SHeRLoc.txt
```
- 4D Radar Place Recognition
```
cd SHeRLoc/training
python train.py --config ../config/config.txt --model_config ../models/4d_SHeRLoc.txt
```
- Spinning Radar Place Recognition
```
cd SHeRLoc/training
python train.py --config ../config/config.txt --model_config ../models/Spinning_SHeRLoc.txt
```
- LiDAR to Spinning Radar Place Recognition
```
cd SHeRLoc/training
python train.py --config ../config/config.txt --model_config ../models/LiDAR_SHeRLoc.txt
```
**Train with the smaller model (SHeRLoc-S):**
- Heterogeneous Radar Place Recognition
```
cd SHeRLoc/training
python train.py --config ../config/config.txt --model_config ../models/SHeRLoc_S.txt
```
- 4D Radar Place Recognition
```
cd SHeRLoc/training
python train.py --config ../config/config.txt --model_config ../models/4d_SHeRLoc_S.txt
```
- Spinning Radar Place Recognition
```
cd SHeRLoc/training
python train.py --config ../config/config.txt --model_config ../models/Spinning_SHeRLoc_S.txt
```
- LiDAR to Spinning Radar Place Recognition
```
cd SHeRLoc/training
python train.py --config ../config/config.txt --model_config ../models/LiDAR_SHeRLoc_S.txt
```

**3-2 Pre-trained Models**

Pre-trained models are available in the `weights` directory:
- `Hetero_SHeRLoc.pth`: Heterogeneous Radar PR Model (320-dimension descriptor)
- `Hetero_SHeRLoc_S.pth`: Heterogeneous Radar PR Model (256-dimension descriptor)
- `4d_SHeRLoc.pth`: 4D Radar PR Model (320-dimension descriptor)
- `4d_SHeRLoc_S.pth`: 4D Radar PR Model (256-dimension descriptor)
- `Spinning_SHeRLoc.pth`: Spinning Radar PR Model (320-dimension descriptor)
- `Spinning_SHeRLoc_S.pth`: Spinning Radar PR Model (256-dimension descriptor)
- `LiDAR_SHeRLoc.pth`: LiDAR to Spinning Radar PR Model (320-dimension descriptor)
- `LiDAR_SHeRLoc_S.pth`: LiDAR to Spinning Radar PR Model (256-dimension descriptor)

**3-3 Evaluation**

To evaluate pretrained models run the following commands.

**For the standard model (SHeRLoc):**
- Heterogeneous Radar Place Recognition
```
cd SHeRLoc/eval
python evaluate.py --dataset_root /code/SHeRLoc/datasets/HeRCULES --map_seq Sports_Complex/01 --query_seq Sports_Complex/01 --model_config ../models/Hetero_SHeRLoc.txt --weights ../weights/Hetero_SHeRLoc.pth
```
- 4D Radar Place Recognition
```
cd SHeRLoc/eval
python evaluate.py --dataset_root /code/SHeRLoc/datasets/HeRCULES --map_seq Sports_Complex/01 --query_seq Sports_Complex/02 --model_config ../models/4d_SHeRLoc.txt --weights ../weights/4d_SHeRLoc.pth
```
- Spinning Radar Place Recognition
```
cd SHeRLoc/eval
python evaluate.py --dataset_root /code/SHeRLoc/datasets/HeRCULES --map_seq Sports_Complex/01 --query_seq Sports_Complex/02 --model_config ../models/Spinning_SHeRLoc.txt --weights ../weights/Spinning_SHeRLoc.pth
```
- LiDAR to Spinning Radar Place Recognition
```
cd SHeRLoc/eval
python evaluate.py --dataset_root /code/SHeRLoc/datasets/HeRCULES --map_seq Sports_Complex/01 --query_seq Sports_Complex/01 --model_config ../models/LiDAR_SHeRLoc.txt --weights ../weights/LiDAR_SHeRLoc.pth
```
**For the smaller model (SHeRLoc-S):**
- Heterogeneous Radar Place Recognition
```
cd SHeRLoc/eval
python evaluate.py --dataset_root /code/SHeRLoc/datasets/HeRCULES --map_seq Sports_Complex/01 --query_seq Sports_Complex/01 --model_config ../models/Hetero_SHeRLoc_S.txt --weights ../weights/Hetero_SHeRLoc_S.pth
```
- 4D Radar Place Recognition
```
cd SHeRLoc/eval
python evaluate.py --dataset_root /code/SHeRLoc/datasets/HeRCULES --map_seq Sports_Complex/01 --query_seq Sports_Complex/02 --model_config ../models/4d_SHeRLoc_S.txt --weights ../weights/4d_SHeRLoc_S.pth
```
- Spinning Radar Place Recognition
```
cd SHeRLoc/eval
python evaluate.py --dataset_root /code/SHeRLoc/datasets/HeRCULES --map_seq Sports_Complex/01 --query_seq Sports_Complex/02 --model_config ../models/Spinning_SHeRLoc_S.txt --weights ../weights/Spinning_SHeRLoc_S.pth
```
- LiDAR to Spinning Radar Place Recognition
```
cd SHeRLoc/eval
python evaluate.py --dataset_root /code/SHeRLoc/datasets/HeRCULES --map_seq Sports_Complex/01 --query_seq Sports_Complex/01 --model_config ../models/LiDAR_SHeRLoc_S.txt --weights ../weights/LiDAR_SHeRLoc_S.pth
```
## License and Citation
- When using the code or dataset, please cite our paper:
```
@article{kim2025sherloc,
  title = {SHeRLoc: Synchronized Heterogeneous Radar Place Recognition for Cross-Modal Localization},
  author = {Kim, Hanjun and Jung, Minwoo and Yang, Wooseong and Kim, Ayoung},
  journal = {IEEE Robotics and Automation Letters},
  year = {2025},
  publisher = {IEEE},
}

@INPROCEEDINGS { hjkim-2025-icra,
    AUTHOR = { Hanjun Kim and Minwoo Jung and Chiyun Noh and Sangwoo Jung and Hyunho Song and Wooseong Yang and Hyesu Jang and Ayoung Kim },
    TITLE = { HeRCULES: Heterogeneous Radar Dataset in Complex Urban Environment for Multi-session Radar SLAM },
    BOOKTITLE = { Proceedings of the IEEE International Conference on Robotics and Automation (ICRA) },
    YEAR = { 2025 },
    MONTH = { May. },
    ADDRESS = { Atlanta },
}
```


## Contributors
- Maintainer: Hanjun Kim (hanjun815@snu.ac.kr)
