## Solution for VisDA 2020 

+ Technique Report: [arXiv](https://arxiv.org/abs/2008.10313) or [GitHub](files/visda.pdf)
+ Video Introduction: [YouTube](https://youtu.be/Ox-ZJhgFwSU) or [bilibili](https://www.bilibili.com/video/BV14V411U7mb)

### Requirements

+ Python 3
+ PyTorch >= 1.1

### Installation

+ Install `visda` library
```shell
git clone https://github.com/yxgeee/MMT-plus.git
cd MMT-plus
python setup.py develop
```
+ Install `NVIDIA/apex` library (**optional**, for mixed precision training)
```shell
git clone https://github.com/NVIDIA/apex.git
cd apex
python setup.py install --cuda_ext --cpp_ext
```
Active it by adding `--fp16` in the training commands.

### Prepare Datasets
`personX_sda` can be downloaded from [[Google Drive]](https://drive.google.com/file/d/1gX_A2AknZp8GtQqgtW2UVwcSCmhFhOgN/view?usp=sharing), while the others can be downloaded from [[Simon4Yan/VisDA2020]](https://github.com/Simon4Yan/VisDA2020#challenge-data).
```
examples
├── data
│   ├── index_validation_gallery.txt  
│   ├── index_validation_query.txt  
│   ├── index_test_gallery.txt  
│   ├── index_test_query.txt     
│   ├── personX
│   ├── personX_sda
│   ├── target_training
│   ├── target_test
└── └── target_validation
```

### Testing

The trained models for our submission can be downloaded from:
+ Re-ID model:
  + ResNeSt-50 [[Google Drive]](https://drive.google.com/file/d/1Wd0SZB_K896rrcmBTOKtvN8U_GlgCfLc/view?usp=sharing)
  + ResNeSt-101 [[Google Drive]](https://drive.google.com/file/d/1U99XtSUxejH_RjdbNShKQKQ4SvleROJA/view?usp=sharing)
  + DenseNet-169-IBN [[Google Drive]](https://drive.google.com/file/d/1oOzYsGzqIC4m9u_dOw4UOJ4d-wlZ9zTA/view?usp=sharing)
  + ResNeXt-101-IBN [[Google Drive]](https://drive.google.com/file/d/1SvflLaRifGtl5hh5kw6ZsVIABjxdEp-w/view?usp=sharing)
+ Camera model:
  + ResNeSt-50 [[Google Drive]](https://drive.google.com/file/d/1qd8ybMtLWXRQOasQzHLh6ZWFLOvwv4Pe/view?usp=sharing)

#### Test a single model
```shell
CUDA_VISIBLE_DEVICES=0 ./scripts/test.sh $ARCH_REID $PATH_REID $ARCH_CAMERA $PATH_CAMERA
```
If you want to test a model without domain-specific BN (e.g. pre-trained model), you need to remove `--dsbn` from `scripts/test.sh`.

#### Model ensembling and testing
```shell
CUDA_VISIBLE_DEVICES=0 ./scripts/test_ensemble.sh
```
Please make sure the model path in `scripts/test_ensemble.sh` is correct before testing.

#### Top-3 results on the leaderboard
| Team Name | mAP(%) | top-1(%) |
| ----- | :------: | :---------: |
| Vimar Team | 76.56 | 84.25 |
| **Ours** | **74.78** | **82.86** |
| Xiangyu | 72.39 | 83.85 |

### Training

<p align="center">
    <img src="files/pipeline.png" width="100%">
</p>

#### Stage I: Structured Domain Adaptation (SDA)

You could directly download the generated images from [[Google Drive]](https://drive.google.com/file/d/1gX_A2AknZp8GtQqgtW2UVwcSCmhFhOgN/view?usp=sharing), or you could use the following scripts to train your own SDA model.

<!-- <p align="center">
    <img src="files/sda.png" width="70%">
</p> -->

+ Train SDA:
```shell
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 ./scripts/train_sda.sh $ARCH $SOUECE_MODEL $TARGET_MODEL
```
+ Generate source-to-target images:
```shell
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 ./scripts/test_sda.sh $PATH_OF_GENERATOR
```

#### Stage II: Pre-training

```shell
CUDA_VISIBLE_DEVICES=0,1,2,3 ./scripts/pretrain.sh personx_sda $ARCH 1
CUDA_VISIBLE_DEVICES=0,1,2,3 ./scripts/pretrain.sh personx_sda $ARCH 2
```

#### Stage III: Improved Mutual Mean-Teaching (MMT+)

<!-- <p align="center">
    <img src="files/mmt+.png" width="100%">
</p> -->

```shell
CUDA_VISIBLE_DEVICES=0,1,2,3 ./scripts/train_mmt_dbscan.sh $ARCH
```

#### Post-processing: Camera Classification Training

```shell
CUDA_VISIBLE_DEVICES=0 ./scripts/camera.sh $ARCH
```
The trained camera model will be used in the inference stage.


### Citation
If you find this code useful for your research, please consider cite:
```
@misc{ge2020improved,
    title={Improved Mutual Mean-Teaching for Unsupervised Domain Adaptive Re-ID},
    author={Yixiao Ge and Shijie Yu and Dapeng Chen},
    year={2020},
    eprint={2008.10313},
    archivePrefix={arXiv},
    primaryClass={cs.CV}
}

@inproceedings{ge2020mutual,
  title={Mutual Mean-Teaching: Pseudo Label Refinery for Unsupervised Domain Adaptation on Person Re-identification},
  author={Yixiao Ge and Dapeng Chen and Hongsheng Li},
  booktitle={International Conference on Learning Representations},
  year={2020},
  url={https://openreview.net/forum?id=rJlnOhVYPS}
}

@misc{ge2020structured,
    title={Structured Domain Adaptation with Online Relation Regularization for Unsupervised Person Re-ID},
    author={Yixiao Ge and Feng Zhu and Rui Zhao and Hongsheng Li},
    year={2020},
    eprint={2003.06650},
    archivePrefix={arXiv},
    primaryClass={cs.CV}
}
```

### Acknowledgement
This code is mainly based on [MMT](https://github.com/yxgeee/MMT).
