
<div align="center">
<h1>PatchRefiner </h1>
<h3>Leveraging Synthetic Data for Real-Domain High-Resolution <br> Monocular Metric Depth Estimation</h3>

[![Paper](https://img.shields.io/badge/arXiv-PDF-b31b1b)](https://arxiv.org/abs/2406.06679) [![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)

<a href="https://zhyever.github.io/">Zhenyu Li</a>, <a href="https://shariqfarooq123.github.io/">Shariq Farooq Bhat</a>, <a href="https://peterwonka.net/">Peter Wonka</a>. 
<br>KAUST

</div>

## ✨ **NEWS**
- 2024-08-15: Release codes and models
- 2024-07-01: Accepted to ECCV 2024.

## **Repo Features**
- 2024-08-15: PatchRefiner repo inherits all features from the [PatchFusion](https://github.com/zhyever/PatchFusion) repo. Please check introductions in PatchFusion repo about basic [training](https://github.com/zhyever/PatchFusion/blob/main/docs/user_training.md), [inference](https://github.com/zhyever/PatchFusion/blob/main/docs/user_infer.md), etc.
 
## **Environment setup**

Install environment using `environment.yml` : 

Using [mamba](https://github.com/mamba-org/mamba) (fastest):
```bash
mamba env create -n patchrefiner --file environment.yml
mamba activate patchrefiner
```
Using conda : 

```bash
conda env create -n patchrefiner --file environment.yml
conda activate patchrefiner
```

### NOTE:
Before running the code, please first run:
```bash
export PYTHONPATH="${PYTHONPATH}:/path/to/the/folder/PatchRefiner"
export PYTHONPATH="${PYTHONPATH}:/path/to/the/folder/PatchRefiner/external"
```
**Make sure that you have exported the `external` folder which stores codes from other repos (ZoeDepth, Depth-Anything, etc.)**

## **Pre-Train Model**

Before training and inference, please prepare some pretrained models from [here](https://drive.google.com/drive/folders/1-ScqRuAIxHPxgubdJzC1CDjmJbQK__mD?usp=sharing).

Unzip the file and make sure you have the `work_dir` folder in this repo after that. 

## **User Inference**

### Running:
To execute user inference, use the following command:

```bash
python tools/test.py ${CONFIG_FILE} --ckp-path <checkpoints> --cai-mode <m1 | m2 | rn> --cfg-option general_dataloader.dataset.rgb_image_dir='<img-directory>' [--save] --work-dir <output-path> --test-type general [--gray-scale] --image-raw-shape [h w] --patch-split-num [h, w]
```
Arguments Explanation (More details can be found [here](https://github.com/zhyever/PatchFusion/blob/main/docs/user_infer.md)):
- `${CONFIG_FILE}`: Select the configuration file from the following options based on the inference type you want to run:
    - `configs/patchrefiner_zoedepth/pr_u4k.py` for PatchRefiner based on ZoeDepth and trained on the Unreal4KDataset (Synthetic Data).
    - `configs/patchrefiner_zoedepth/pr_cs.py` for PatchRefiner based on ZoeDepth and trained on the Unreal4KDataset (Synthetic Data) and CityScapesDataset (Real Data).
- `--ckp-path`: Specify the checkpoint path.
    - `work_dir/zoedepth/cs/pr/checkpoint_05.pth` for PatchRefiner based on ZoeDepth and trained on the Unreal4KDataset.
    - `work_dir/zoedepth/cs/ssi_7e-2/checkpoint_02.pth` for PatchRefiner based on ZoeDepth and trained on the Unreal4KDataset and CityScapesDataset (Real Data).
- `--cai-mode`: Define the specific mode to use. For example, rn indicates n patches in mode r.
- `--cfg-option`: Specify the input image directory. Maintain the prefix as it indexes the configuration. (To learn more about this, please refer to [MMEngine](https://mmengine.readthedocs.io/en/latest/advanced_tutorials/config.html). Basically, we use MMEngine to organize the configurations of this repo).
- `--save`: Enable the saving of output files to the specified `--work-dir` directory (Make sure using it, otherwise there will be nothing saved).
- `--work-dir`: Directory where the output files will be stored, including a colored depth map and a 16-bit PNG file (multiplier=256).
- `--gray-scale`: If set, the output will be a grayscale depth map. If omitted, a color palette is applied to the depth map by default.
- `--image-raw-shape`: Specify the original dimensions of the input image. Input images will be resized to this resolution before being processed by the model. Default: `2160 3840`.
- `--patch-split-num`: Define how the input image is divided into smaller patches for processing. Default: `4 4`. ([Check more introductions](https://github.com/zhyever/PatchFusion/blob/main/docs/user_infer.md))

### Example Usage:
Below is an example command that demonstrates how to run the inference process:
```bash
python ./tools/test.py configs/patchrefiner_zoedepth/pr_u4k.py --ckp-path work_dir/zoedepth/cs/pr/checkpoint_05.pth --cai-mode r32 --cfg-option general_dataloader.dataset.rgb_image_dir='./examples/' --save --work-dir ./work_dir/predictions --test-type general --image-raw-shape 1080 1920 --patch-split-num 2 2
```
This example performs inference using the `pr_u4k.py` configuration, loads the specified checkpoint `work_dir/zoedepth/cs/pr/checkpoint_05.pth`, sets the PatchRefiner mode to `r32`, specifies the input image directory `./examples/`, and saves the output to ./work_dir/predictions `./work_dir/predictions`. The original dimensions of the input image is `1080x1920` and the input image is divided into `2x2` patches.

## **User Training**

### Please refer to [user_training](./docs/user_training.md) for more details.

## Citation
If you find our work useful for your research, please consider citing the paper
```
@article{li2024patchrefiner,
    title={PatchRefiner: Leveraging Synthetic Data for Real-Domain High-Resolution Monocular Metric Depth Estimation}, 
    author={Zhenyu Li and Shariq Farooq Bhat and Peter Wonka},
    booktitle={ECCV},
    year={2024}
}
```
