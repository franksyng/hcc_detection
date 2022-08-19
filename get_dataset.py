import cv2
import os
from PIL import Image
from tqdm import tqdm
from utils import check_folder_existence
import argparse
import random


def get_args():
    parser = argparse.ArgumentParser('data_generator')
    parser.add_argument('--root')
    parser.add_argument('--out-dir')
    parser.add_argument('--mask', default=False, type=bool)
    parser.add_argument('--noise', default=False, type=bool)
    parser.add_argument('--vol', type=int, default=4000, help='How many images will be extracted for each mutation and time combination')
    args = parser.parse_args()
    return args


# A function for project development to test whether noises increase model robustness
# The code is borrowed from the internet
# We are not using this function at current version of the project
def add_noise(img):
    # Getting the dimensions of the image
    row , col, _ = img.shape
    # Randomly pick some pixels in the
    # image for coloring them white
    # Pick a random number between 300 and 10000
    number_of_pixels = random.randint(200, 300)
    for i in range(number_of_pixels):
        # Pick a random y coordinate
        y_coord=random.randint(0, row - 1)
        # Pick a random x coordinate
        x_coord=random.randint(0, col - 1)
        # Color that pixel to white
        img[y_coord][x_coord] = 255
    # Randomly pick some pixels in
    # the image for coloring them black
    # Pick a random number between 300 and 600
    number_of_pixels = random.randint(500 , 600)
    for i in range(number_of_pixels):
        # Pick a random y coordinate
        y_coord=random.randint(0, row - 1)
        # Pick a random x coordinate
        x_coord=random.randint(0, col - 1)
        # Color that pixel to black
        img[y_coord][x_coord] = 0
    return img


def image2dataset(img_path, out_dir, use_mask, noise):
    img_num = img_path.split('/')[-1].split('_')[0]
    folder_name = img_path.split('/')[1]

    original_im = cv2.imread(img_path)
    original_im = cv2.cvtColor(original_im, cv2.COLOR_BGR2RGB)
    original_im = cv2.GaussianBlur(original_im, (3, 3), cv2.BORDER_DEFAULT)
    if noise:
        original_im = add_noise(original_im)

    if use_mask:
        # please change this to a more flexible expression if the mask is no longer named 'X_mask.png'
        mask_path = os.path.join(img_path[:-10] + '_mask.png')
        mask = cv2.imread(mask_path)
        mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
        res = cv2.bitwise_and(original_im, original_im, mask=mask)
        res_im = Image.fromarray(res)
        res_im.save(os.path.join(out_dir, folder_name + '_' + img_num + '.png'))
    else:
        res_im = Image.fromarray(original_im)
        res_im.save(os.path.join(out_dir, folder_name + '_' + img_num + '.png'))


def generate_mask_data(args):
    root_dir = args.root
    out_dir = args.out_dir
    use_mask = args.mask
    noise = args.noise
    vol = args.vol
    dirs = os.listdir(root_dir)
    check_folder_existence(out_dir)
    
    group_in_id = {}
    for each_dir in dirs:
        if each_dir != '.DS_Store':
            dir_path = os.path.join(root_dir, each_dir)
            mouse_id = each_dir.split('_')[0]
            imgs = os.listdir(dir_path)
            img_in_dir = []
            for img in imgs:
                img_path = os.path.join(dir_path, img)
                img_in_dir.append(img_path)
            random.shuffle(img_in_dir)
            if mouse_id in group_in_id:
                group_in_id[mouse_id].extend(img_in_dir)
            else:
                group_in_id[mouse_id] = img_in_dir
        else:
            continue

    for mouse in group_in_id:
        imgs = group_in_id[mouse]
        random.shuffle(imgs)
        counter = 0
        for img in tqdm(imgs):
            if img !='.DS_Store' and 'image' in img and counter < vol:
                image2dataset(img, out_dir, use_mask, noise)
                counter += 1


if __name__ == '__main__':
    args = get_args()
    generate_mask_data(args)


