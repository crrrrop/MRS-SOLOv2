# MRS-SOLOv2: Soybean Pod Instance Segmentation and Phenotypic Trait Calculation

This repository provides the implementation of **MRS-SOLOv2**, an improved SOLOv2-based instance segmentation model for soybean pod recognition and phenotypic trait calculation from RGB images.

The model is designed for in situ soybean pod images with dense pod distribution, mutual occlusion, boundary adhesion, similar pod-background color, and large variation in pod size. The predicted instance masks can be used for downstream pod phenotypic trait calculation, including pod length, width, area, and seed-number-related traits.

## Main Features

- Improved SOLOv2 framework for soybean pod instance segmentation.
- ResNeSt-based backbone for enhanced discriminative feature extraction.
- MSCA-FPN-based multi-scale contextual feature fusion.
- COCO-style dataset support for training and evaluation.
- Evaluation with AP, mAP@0.5, mAP@0.75, parameter count, FPS, and peak GPU memory.
- Demo scripts for soybean pod mask prediction and visualization.

## Repository Structure

```text
MRS-SOLOv2/
├── config/                 # Model and training configuration files
├── data/                   # Class name files
├── dataset/                # Dataset placeholder; not included in this repository
├── model/                  # Network modules, backbone, FPN, heads, losses, decoder
├── tools/                  # Data processing, transforms, COCO evaluation, visualization
├── train.py                # Training entry
├── eval.py                 # Evaluation entry
├── demo.py                 # General inference demo
├── demo_pod.py             # Soybean pod inference demo
├── getparameters.py        # Parameter counting utility
└── README.md
```

## Environment

The code was developed with PaddlePaddle. The experimental environment used in the manuscript was based on PaddlePaddle-GPU.

Recommended environment:

```text
Python >= 3.7
PaddlePaddle-GPU 2.0.0rc0 or a compatible PaddlePaddle-GPU version
CUDA-compatible NVIDIA GPU
pycocotools
opencv-python
numpy
```

Install common dependencies:

```bash
pip install numpy opencv-python pycocotools
```

Install PaddlePaddle-GPU according to your CUDA version from the official PaddlePaddle installation guide.

## Dataset Preparation

The dataset is not included in this repository. Please prepare the soybean pod dataset in COCO format and place it as follows:

```text
dataset/
└── line/
    ├── annotations/
    │   ├── instances_line_train.json
    │   └── instance_line_val.json
    ├── train/
    │   ├── image_001.jpg
    │   └── ...
    └── val/
        ├── image_001.jpg
        └── ...
```

The default dataset paths are defined in:

```text
config/solov2_rs50_fpn_8gpu_3x.py
```

Default paths:

```python
train_path = 'dataset/line/annotations/instances_line_train.json'
val_path = 'dataset/line/annotations/instance_line_val.json'
train_pre_path = 'dataset/line/train/'
val_pre_path = 'dataset/line/val/'
classes_path = 'data/pod_classes.txt'
```

If your dataset is stored elsewhere, update these paths in the configuration file.

## Model Weights

Large model weights are not included in this repository. Please download the trained weights from the release page or the external link provided by the authors, and place them in the path expected by the configuration file:

```text
weights_St+MSCA34/best_model.pdparams
```

Alternatively, update `model_path` in `config/solov2_rs50_fpn_8gpu_3x.py`:

```python
eval_cfg = dict(
    model_path='weights_St+MSCA34/best_model.pdparams',
    ...
)

test_cfg = dict(
    model_path='weights_St+MSCA34/best_model.pdparams',
    ...
)
```

## Configuration Selection

The command-line argument `--config` selects the configuration file:

| Value | Configuration file |
|---:|---|
| 0 | `config/solov2_r50_fpn_8gpu_3x.py` |
| 1 | `config/solov2_light_448_r50_fpn_8gpu_3x.py` |
| 2 | `config/solov2_light_r50_vd_fpn_dcn_512_3x.py` |
| 3 | `config/solov2_rs50_fpn_8gpu_3x.py` |

The default configuration is `--config 3`, which corresponds to the MRS-SOLOv2 setting used in the manuscript.

## Training

Run training with GPU:

```bash
python train.py --use_gpu True --config 3
```

Run training with CPU:

```bash
python train.py --use_gpu False --config 3
```

Main training settings in the default configuration:

```text
Batch size: 2
Maximum iterations: 42000
Optimizer: Momentum
Initial learning rate: 0.01
Momentum: 0.9
Weight decay: 0.0001
Warm-up steps: 1000
Input multi-scale target sizes: 640, 672, 704, 736, 768, 800
Maximum image size: 1333
Random horizontal flipping probability: 0.5
```

## Evaluation

Evaluate the model on the validation set:

```bash
python eval.py --use_gpu True --config 3
```

The evaluation script uses the validation annotation file and image folder defined in the selected configuration file.

## Inference Demo

Before running the demo, place test images in:

```text
images/test/
```

Run soybean pod inference:

```bash
python demo_pod.py --use_gpu True --config 3
```

The default output folders are:

```text
images/res/
images/masks/
images/colored_masks/
```

These folders are generated outputs and should not be committed to GitHub.

## Reported Performance

The manuscript reports the following performance for MRS-SOLOv2 on the soybean pod dataset:

| Model | AP | mAP@0.5 | mAP@0.75 | Parameters/M | FPS | Peak GPU memory/GB |
|---|---:|---:|---:|---:|---:|---:|
| SOLOv2 | 0.622 | 0.735 | 0.663 | 46.29 | 9.2 | 3.5 |
| MRS-SOLOv2 | 0.697 | 0.818 | 0.724 | 46.51 | 9.7 | 4.2 |

The phenotypic calculation results reported in the manuscript achieved R2 values of 0.9549, 0.9498, 0.9585, and 0.9742 for pod length, pod width, pod area, and seed-number-related traits, respectively.

## Files Not Included

The following files and folders are intentionally excluded from the GitHub repository:

```text
runs/
dataset image files and annotations, unless separately released
images/
eval_results/
*.pdparams
*.pdopt
*.pdmodel
*.pdiparams
__pycache__/
*.pyc
.ipynb_checkpoints/
```

Datasets and trained weights should be released separately, for example through GitHub Releases, Zenodo, Figshare, or another data repository.

## Citation

If this code is useful for your research, please cite the corresponding manuscript:

```text
[Authors]. Soybean pod phenotypic trait recognition and calculation using an improved instance segmentation model. [Journal/Year].
```

## License

Please add a license before public release, for example MIT, Apache-2.0, or another license approved by your institution.
