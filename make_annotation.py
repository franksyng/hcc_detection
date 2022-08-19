from torchvision import transforms
import numpy as np
import cv2
import os
from PIL import Image
from tqdm import tqdm
from utils import make_annotation
import argparse


# dataset = GOFandLOF(annotations='annotations.csv', img_dir='converted_img', transform=None)
# image_dir = 'testset/1024455_Kras_P53_21d_IHC_00008_JPG'

# image_dir = 'train_val/104450_Bcat_P53_7d_IHC_00000_JPG'


def get_args():
    parser = argparse.ArgumentParser('annotation_maker')

    parser.add_argument('--data')
    parser.add_argument('--multitask', default=False, type=bool)
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = get_args()
    dataset = args.data
    make_annotation(dataset)

