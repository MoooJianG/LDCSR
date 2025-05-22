
<div align="center">

<h1>Latent Diffusion for Continuous-Scale Super-Resolution of Remote-Sensing Images
</h1>

<div>
    <a href='https://hlwu.ac.cn/' target='_blank'>Hanlin Wu</a>&emsp;
    Jiangwei Mo&emsp;
    Xiaohui Sun&emsp;
    Jie Ma
</div>
<div>
    Beijing Foreign Studies University
</div>

[**Paper**](https://arxiv.org/abs/2410.22830) | [**PDF**](https://arxiv.org/pdf/2410.22830)

---

</div>


## Overview
![overview](asserts/flowchart.png)

<!-- ## Key Highlights -->

## Dependencies and Installation
1. Clone repo
```bash
git clone https://github.com/MoooJianG/LDCSR.git
```
2. Install dependencies
```bash
conda create -n LDCSR python=3.10
conda activate LDCSR
pip install -r requirements.txt
```
## Usage
### Dataset Preparation
We support AID, DOTA, and DIOR out‑of‑the‑box. Download HR images using the official links below (or your own data) and generate LR/HR pairs via bicubic down‑sampling.

| Data Type | [AID]() | [DOTA]() | [DIOR]() |
| :----: | :----: | :----: | :----: |
| HR | [HR]() | None | None |
| LR | [LR]() | [LR]() | [LR]() |

```bash
# Example: split AID
python data/prepare_split.py --split_file AID_split.pkl --data_path dataset/RawAID --output_path dataset/AID
```
Custom datasets should replicate the following folder structure:
```
└── dataset
    └── YourData
        ├── Train
        |   ├── HR
        |   └── LR
        ├── Test
        └── Val
```

### Quick Start
#### Model Training in AID
```bash
# First-stage
python train.py --config configs/first_stage_kl_v6.yaml
# Second-stage
python train.py --config configs/second_stage_van_v4.yaml
```
#### Model Testing
```bash
python test.py --checkpoint path/to/checkpoint.ckpt --datasets AID --scales 4
```


## Contact
If you have any questions or suggestions, feel free to contact me.

Email：20220119004@bfsu.edu.cn
## Citation
```
@ARTICLE{11006698,
  author={Wu, Hanlin and Mo, Jiangwei and Sun, Xiaohui and Ma, Jie},
  journal={IEEE Transactions on Geoscience and Remote Sensing}, 
  title={Latent Diffusion, Implicit Amplification: Efficient Continuous-Scale Super-Resolution for Remote Sensing Images}, 
  year={2025},
  volume={},
  number={},
  pages={1-1},
  keywords={Diffusion models;Training;Image synthesis;Noise reduction;Visualization;Decoding;Computational modeling;Remote sensing;Image reconstruction;Autoencoders;Remote sensing;super-resolution;latent diffusion;continuous-scale},
  doi={10.1109/TGRS.2025.3571290}}

```
<!-- ## License -->
