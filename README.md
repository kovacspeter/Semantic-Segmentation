# Semantic-Segmentation
Implementation of various models for Semantic Segmentation.

### Currently supported segmentation models

- [ ] FCN
- [x] Segnet_VGG16
- [ ] Segnet with different pretrained networks

# Installation
`pip install -r requirements.txt`

### Data
Currently supported loaders are only VOC PASCAL.

You can find all data here:

[PASCAL VOC](http://host.robots.ox.ac.uk/pascal/VOC/)

[SBD](http://home.bharathh.info/pubs/codes/SBD/download.html)

### Config
Dataloader requires path to data. This can be provided through `config.yaml`.

# Training
You can for example run - batch of size 6 takes approximately 9GB of gpu memory

`python train.py --n_epoch 30 --batch_size 6`




### Credits
This repo was inspired by various other repositories and some parts of code
come directly from them. So I list them all here: (Thank you)

[https://github.com/bodokaiser/piwise/tree/master/piwise](https://github.com/bodokaiser/piwise/tree/master/piwise)

[https://github.com/meetshah1995/pytorch-semseg](https://github.com/meetshah1995/pytorch-semseg)