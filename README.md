# HCC_Detection

---

## Label Format
Labels of each image are given in lists [Kras, Bcat] and [7d, 21d, 9w].

In classification output:
1. The classification output files are named as: GOF_LOF_TIME_ImageID_image.png

## Models
- [x] ResNet50(baseline)
- [x] ResNet50(2-layer MLP)
- [x] ResNet50(3-layer MLP)
- [x] ResNet50(bottleneck)
- [x] ResNet101

### Training
To train with RGB image:

```python train.py```

To train in grayscale:

```python train.py --grayscale```

### Evaluation
To evaluate model:

```python eval.py --testset [testset_path] --gt [ground_truth] --cls-target [class_target]```

To evaluate and generate saliency maps:

```python eval.py --saliency ['full'/'grad'] --testset [testset_path] --gt [ground_truth] --cls-target [class_target] --grayscale```

For detailed evaluation parameter instructions, check customisable evaluation parameters below.

## Customisable Training Parameters

    --arch: default='R50Baseline', 'Specify one model architecture'

    --data: default='data', 'Specify data folder'
    --train-percent: default=0.8
    --cross-val: default=None, 'Input fold number to enable cross val' # Not available
    
    --epoch: default=10
    --batch-size: default=32
    --lr: default=0.01
    --rm-norm: default=False, 'Disable image normalisation'
    --grayscale: default=False, 'Enable grayscale'
    
    --criterion: default='BCELoss'
    --optimizer: default='SGD'
    --scheduler: default='CosineAnnealingLR'

    --interval: default=10, 'Interval of iteration for printing log in each epoch'


## Customisable Evaluation Parameters

    --arch: default='R50Baseline', 'Specify one model architecture'
    --ckpt: default='ckpt/R50Baseline/best.pth', 'Path of checkpoint file'
    --testset: default='Specify test set folder'
    --gt: 'Provide ground truth label in format GOF_LOF_TIME'
    --cls-target: default=0, 'Specify the class target of activation'
    --saliency: default=False, 'Enable saliency map'
    --rm-norm: default=False, 'Disable image normalisation'
    --grayscale: default=False, 'Enable grayscale'