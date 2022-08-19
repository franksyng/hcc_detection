# HCC_Detection
MSc project: Detecting Oncogenic Mutations and Morphological Changes in Histology Images of Early Cancer via Deep Learning by ©Frank S.Y. Ng

Contact info: franksyng@gmail.com


## Environment Setup
1. Recommend running with python 3.9.
2. Run ```pip install -r requirements.txt``` to install packages.
3. IMPORTANT: Since the output of grad-cam package adds heatmap and original image together by default, either of the following steps should be done or the project will not able to be run. 
   - If not remanding a pure heatmap, please comment the code in line 124 of ```utils.py``` replace it with the code in line 125. 
   - If want to obtain pure heatmap, please replace the function ```show_cam_on_image``` with the following codes. (hints: to find ```show_cam_on_image``` please refer to ```utils.get_saliency``` line 124 and use ```cmd + left click``` to find the usage in package.) 
```
def show_cam_on_image(img: np.ndarray,
                  mask: np.ndarray,
                  use_rgb: bool = False,
                  colormap: int = cv2.COLORMAP_JET,
                  image_weight: float = 0.5) -> np.ndarray:
""" This function overlays the cam mask on the image as an heatmap.
By default the heatmap is in BGR format.

:param img: The base image in RGB or BGR format.
:param mask: The cam mask.
:param use_rgb: Whether to use an RGB or BGR heatmap, this should be set to True if 'img' is in RGB format.
:param colormap: The OpenCV colormap to be used.
:param image_weight: The final result is image_weight * img + (1-image_weight) * mask.
:returns: The default image with the cam overlay.
"""
heatmap = cv2.applyColorMap(np.uint8(255 * mask), colormap)
if use_rgb:
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
heatmap = np.float32(heatmap) / 255

if np.max(img) > 1:
    raise Exception(
        "The input image should np.float32 in the range [0, 1]")

if image_weight < 0 or image_weight > 1:
    raise Exception(
        f"image_weight should be in the range [0, 1].\
            Got: {image_weight}")

cam = (1-image_weight) * heatmap + image_weight * img
cam = cam / np.max(cam)
heatmap = heatmap / np.max(heatmap)
return np.uint8(255 * cam), np.uint8(255 * heatmap)
```
4. Please directly replace the file ```base_cam.py``` in the grad-cam package with the one in this repository. Otherwise, it is unable to get class activation of onset time classification (explanation given in line 77 and 78 of ```base_cam.py```). The class activation of mutation classification will not be affected.

## Models and their class name
- [x] ResNet50(baseline): ResNet50
- [x] ResNet50(2-layer MLP): ResNet50_MLP_2
- [x] ResNet50(3-layer MLP): ResNet50_MLP_3
- [x] ResNet50(bottleneck): ResNet50_BN
- [x] ResNet101: ResNet101

## Prepare dataset

### Extract data

To extract image from the dataset:

```python get_dataset.py --root [data_folder] --out-dir [output_folder]  --vol [volume_for_each_combination]```

e.g. If want to extract 4000 images for each combination in fold1, and the output folder called fold1_4k:

```python get_dataset.py --root fold1 --out-dir fold1_4k  --vol 4000```

### Make annotation

```python make_annotation.py --data [folder_with_extracted_images]```

e.g. Following the get dataset example above:

```python make_annotation.py --data fold1_4k```

## Training

To train in grayscale:

```python train.py --arch [model_class_name] --grayscale --data [data_folder] --annotations [annotation_file.csv]```

e.g. To train with ResNet50_BN in grayscale with annotation file named 'annotations.csv':

```python train.py --arch ResNet50_BN --grayscale --data [data_folder] --annotations annotations.csv```

### Customisable Training Parameters

    --arch: default='ResNet50', 'Specify one model architecture'
    --data: default='data', 'Specify data folder'
    --train-percent: default=0.6
    --epoch: default=50
    --batch-size: default=64
    --lr: default=0.01
    --grayscale: default=False, 'Enable grayscale'
    --loss-ratio: default=0.3, 'Ratio of gene and time in combine loss computation'
    --interval: default=10, 'Interval of iteration for printing log in each epoch'


## Evaluation
### To evaluate images in single tile:

If no saliency map required:

```python eval.py --testset [testset_path] --gt [ground_truth] --cls-target [class_target]```

If requires saliency map:

```python eval.py --testset [testset_path] --gt [ground_truth] --cls-target [class_target] --grayscale --saliency ['full'/'grad']```

### To evaluate a fold and output result csv for each tile:

If no saliency map required:

```python eval_fold.py --arch [model_class_name] --ckpt [ckpt_path] --fold-path [fold_path] --grayscale```

If requires saliency map:

```python eval_fold.py --arch [model_class_name] --ckpt [ckpt_path] --fold-path [fold_path] --grayscale --cls-type [mutation/time] --saliency [full/grad]```

### Customisable Evaluation Parameters

#### For eval.py

    --arch: default='ResNet50', 'Specify one model architecture'
    --ckpt: default='ckpt/ResNet50/best.pth', 'Path of checkpoint file'
    --testset: default='Specify test set folder'
    --gt: 'Provide ground truth label in format GOF_LOF_TIME'
    --cls-target: default=0, 'Specify the class target of activation'
    --saliency: default=False, 'Enable saliency map'
    --grayscale: default=False, 'Enable grayscale'

#### For eval_fold.py

    --arch: default='ResNet50', 'Specify one model architecture'
    --ckpt: default='ckpt/ResNet50/best.pth', 'Path of checkpoint file'
    --fold-path: 'Path to the fold'
    --cls-type: default='mutation', 'The desired type of cls category (mutation/time)'
    --saliency: default=None, 'Enable saliency map'
    --grayscale: default=False, 'Enable grayscale'