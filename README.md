# Solution for VisDA 2020

[Technique Report](visda.pdf)

## Requirements

+ Python 3.*
+ PyTorch >= 1.1

#### Installation

```shell
python setup.py install
```

#### Nvidia Apex (optional, for mixed precision training)
```shell
git clone https://github.com/NVIDIA/apex.git
cd apex
python setup.py install --user # cuda 9.0
python setup.py install --cuda_ext --cpp_ext --user # cuda 10.0
```
Active mixed precision training by adding `--fp16` in the training commands.

#### Prepare Datasets
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


## Training

#### Stage I: SDA

You could directly download the generated images from [Google Drive](https://drive.google.com/file/d/1gX_A2AknZp8GtQqgtW2UVwcSCmhFhOgN/view?usp=sharing), or you could use the following scripts to train your own SDA model.

+ Train SDA:
```shell
./scripts/train_sda.sh $PARTITION
```
+ Generate source-to-target images:
```shell
./scripts/test_sda.sh $PATH $PARTITION
```

#### Stage II: Pre-training

```shell
./scripts/pretrain.sh $ARCH 1 $PARTITION
./scripts/pretrain.sh $ARCH 2 $PARTITION # train twice for init MMT
```

#### Stage III: MMT+ training

```shell
./scripts/train_mmt_dbscan.sh $ARCH $PARTITION
```

#### Testing

+ Test a single model:
```shell
./scripts/test.sh $ARCH $PATH $PARTITION
```
+ Model ensembling and testing:
```shell
./scripts/test_ensemble.sh $PARTITION
```
