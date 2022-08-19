import os
import csv
import cv2
import torch
from PIL import Image
import numpy as np
from torchvision.transforms import transforms

from pytorch_grad_cam import GradCAMPlusPlus, FullGrad, GradCAM, AblationCAM, EigenCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image

# Train with GPU
# The mps is for M1 chip Mac only. If not working, comment it and use the code in line 24.
device = torch.device("cuda:0" if torch.cuda.is_available() else ("mps" if torch.has_mps else "cpu"))
# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


# Temporary function
def gene_pred_to_text(model_output):
    """
    Convert tensor result to text
    :param model_output: a tensor output by the model
    :return: GOF and LOF classification result in text
    """
    probs = model_output.tolist()[0]
    pred = [round(prob) for prob in probs]
    if pred[0] == 1 and pred[1] == 0:
        res = 'Kras'
    elif pred[0] == 0 and pred[1] == 1:
        res = 'Bcat'
    else:
        res = 'Error'
    return res


def pred_to_text(pred):
    """
    Convert tensor result to text
    :param model_output: a tensor output by the model
    :return: GOF, LOF and onset time classification result in text
    """

    if pred[0] == 1:
        gof = 'Kras'
    elif pred[1] == 1:
        gof = 'Bcat'
    else:
        gof = 'Error'

    if pred[2] == 1:
        time = '7d'
    elif pred[3] == 1:
        time = '21d'
    else:
        time = '9w'

    res = gof + '_P53_' + time
    return res


def load_img(img_path, grayscale: bool, is_eval: bool):
    """
    IMPORTANT: please remember cv2 will load and save image in BGR format instead of RGB
    Therefore, if want to save image in RGB, please use PIL
    """
    if not is_eval or 'mask' not in img_path:
        img = cv2.imread(img_path)  # load img
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # convert to RGB
        normalise = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # create a normalisation object
        img = apply_clahe(img, 3.0, (8, 8))  # apply clahe
        img = cv2.GaussianBlur(img, (3, 3), cv2.BORDER_DEFAULT)  # apply gaussian blur for noise reduction
        img = torch.from_numpy(img)  # ndarray -> torch tensor
        img = torch.permute(img, (2, 0, 1))  # switch dimension from ndarray (h, w, channels) to tensor (c, h, w)

        if grayscale:
            # transform to grayscale (it should be kept in 3 channels otherwise will not be able to load in the network)
            transform = transforms.Grayscale(num_output_channels=3)
            img = transform(img)

        if is_eval:
            # if it is doing evaluation, unsqueeze for one more dimension (batchsize, c, h, w)
            # since we evaluate image one by one, the dimension is (1, c, h, w)
            img = torch.unsqueeze(img, dim=0)

        img = img.to(device, dtype=torch.float32)
        img = normalise(img)  # do normalisation
        return img


def get_saliency_map(model, target_layers, cls_target, img_path, cam_type: str = None, grayscale: bool = False):
    """
    The original version of GradCAM does not support the output of heatmap itself. A modification is needed for the
    show_cam_on_image function.
    """
    model.eval()

    if torch.cuda.is_available():
        if cam_type == 'grad':
            cam = GradCAM(model=model, target_layers=target_layers, use_cuda=True)
        elif cam_type == 'full':
            cam = FullGrad(model, target_layers=target_layers, use_cuda=True)
        else:
            print('No CAM type selected')
    else:
        if cam_type == 'grad':
            cam = GradCAM(model=model, target_layers=target_layers, use_cuda=False)
        elif cam_type == 'full':
            cam = FullGrad(model, target_layers=target_layers, use_cuda=False)
        else:
            print('No CAM type selected')

    img = load_img(img_path, grayscale, True)

    pred_gene, pred_time = model(img)
    targets = [ClassifierOutputTarget(cls_target)]  # Kras: 0, Bcat: 1, 7d: 3, 21d: 4, 9w: 5 (i.e., the index in label)
    grayscale_cam = cam(input_tensor=img, targets=targets)
    original_img = cv2.imread(img_path)
    original_img = cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB).astype(np.float32)/255.0
    grayscale_cam = np.transpose(grayscale_cam, (1, 2, 0))

    # for the reason mentioned at the beginning of this function,
    # if the function is unable to run, comment the line of code below and use the next line
    heatmap_with_img, heatmap = show_cam_on_image(original_img, grayscale_cam, use_rgb=True, colormap=cv2.COLORMAP_INFERNO)
    # heatmap_with_img = show_cam_on_image(original_img, grayscale_cam, use_rgb=True)
    pred = res_in_binary(torch.softmax(pred_gene, dim=1))[0] + res_in_binary(torch.softmax(pred_time, dim=1))[0]
    res = pred_to_text(pred)
    return heatmap, heatmap_with_img, res


# unused function because masking is not implemented for current version
def apply_mask(img):
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    ret, th = cv2.threshold(blur, 0, 200, cv2.THRESH_BINARY + cv2.THRESH_OTSU)  # get binary threshold
    th_inv = cv2.bitwise_not(th)  # generate mask
    res = cv2.bitwise_and(img, img, mask=th_inv)  # apply mask
    # res = cv2.addWeighted(img, 0.3, res, 0.7, 0)
    return res


def apply_clahe(image, clip_limit, tile_grid_size):
    """
    For explanations please refers to:
    https://www.geeksforgeeks.org/clahe-histogram-eqalization-opencv/
    """
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    lab_img = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab_img)  # extract the L channel for intensity
    equ_l = clahe.apply(l)  # do equalisation
    equ_img = cv2.merge((equ_l, a, b))  # merge back to LAB image
    equ_img = cv2.cvtColor(equ_img, cv2.COLOR_LAB2RGB)  # convert LAB to RGB
    return equ_img


def check_folder_existence(dir_path):
    """
    Check folder exist or not. If not, create one.
    """
    if not os.path.exists(dir_path):
        os.makedirs(dir_path, exist_ok=True)


def save_model_state(model, ckpt_dir, epoch_num, best_epoch, best_only=False):
    check_folder_existence(ckpt_dir)
    if not best_only:
        # save each checkpoint
        torch.save(model.state_dict(), os.path.join(ckpt_dir, str(epoch_num) + '.pth'))
    if epoch_num == best_epoch:
        # save and replace the best checkpoint
        torch.save(model.state_dict(), os.path.join(ckpt_dir, 'best.pth'))


def res_in_binary(output):
    """
    Transform model prediction and label tensor to a one-hot list for accuracy computation
    """
    return [[round(prob.tolist()) for prob in each_iter] for each_iter in output]


def make_annotation(data_folder):
    """
    Label form in onehot: [Kras, Bcat, P53, 7d, 21d, 9w]
    """
    imgs = os.listdir(data_folder)
    with open('annotations.csv', 'w', encoding='UTF-8') as f:
        f.close()

    for img in imgs:
        if img == '.DS_Store':
            continue
        annotation = img.split('_')
        gof = annotation[1]
        lof = annotation[2]
        time = annotation[3]
        labels = [0, 0, 1, 0, 0, 0]  # P53 set to 1 because of the preliminary test

        # Make gene labels
        if gof == 'Kras':
            labels[0] = 1
        if gof == 'Bcat':
            labels[1] = 1
        if lof == 'P53':
            labels[2] = 1

        # Make time labels
        if time == '7d':
            labels[3] = 1
        elif time == '21d':
            labels[4] = 1
        else:
            labels[5] = 1

        # Write annotations
        with open('ann_' + data_folder + '.csv', 'a', encoding='UTF-8') as f:
            writer = csv.writer(f)
            writer.writerow([img, labels[0], labels[1], labels[2], labels[3], labels[4], labels[5]])
    f.close()


# a function during project development to recursively convert .tiff to .jpg
# not being used for now
def recursively_convert_to_jpg(data_folder, out_path='converted_img'):
    check_folder_existence(out_path)
    folders = os.listdir(data_folder)
    for folder in folders:
        if folder == '.DS_Store':
            continue
        img_path = os.path.join(data_folder, folder)
        imgs = os.listdir(img_path)
        for img in imgs:
            if img == '.DS_Store':
                continue
            img_name = img.split('.')
            tif_img = Image.open(os.path.join(img_path, img))
            output_folder = os.path.join(out_path)
            tif_img.save(os.path.join(output_folder, img_name[0] + '.jpg'))


# a function during project development to convert .tiff to .jpg
# not being used for now
def convert_to_jpg(data_folder, out_path='converted_img'):
    check_folder_existence(out_path)
    imgs = os.listdir(data_folder)
    for img in imgs:
        if img == '.DS_Store':
            continue
        img_name = img.split('.')
        tif_img = Image.open(os.path.join(data_folder, img))
        output_folder = os.path.join(out_path)
        tif_img.save(os.path.join(output_folder, img_name[0] + '.jpg'))

