import os
import argparse
import torch
import utils
from all_models import *
from tqdm import tqdm
from PIL import Image
from sys import stdout
import csv

# GPU
device = torch.device("cuda:0" if torch.cuda.is_available() else ("mps" if torch.has_mps else "cpu"))
# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print('Using', device)


def get_args():
    parser = argparse.ArgumentParser('eval')

    parser.add_argument('--arch', default='ResNet50', help='Specify one model architecture')
    parser.add_argument('--ckpt', default='ckpt/ResNet50/best.pth', help='Choose checkpoint file')
    parser.add_argument('--testset', help='Specify test set folder')
    parser.add_argument('--gt', help='Provide ground truth label in format GOF_LOF_TIME')
    parser.add_argument('--cls-target', default=0, type=int, help='Specify the class target of activation')
    parser.add_argument('--saliency', default=None, type=str, help='Enable saliency map')
    parser.add_argument('--grayscale', default=False, type=bool, action=argparse.BooleanOptionalAction, help='Enable grayscale')
    parser.add_argument('--eval-as-fold', default=False, type=bool, action=argparse.BooleanOptionalAction, help='Eval all folder in one fold')
    args = parser.parse_args()
    return args


def evaluation(args):
    mdl_name = args.arch
    ckpt_path = args.ckpt
    testset = args.testset
    gt = args.gt
    cam_type = args.saliency
    grayscale = args.grayscale
    eval_as_fold = args.eval_as_fold
    imgs = os.listdir(testset)
    correct_num = 0
    total_num = len(imgs)
    if mdl_name == 'ResNet50':
        model = ResNet50()
        target_layers = [model.relu]
    elif mdl_name == 'ResNet101':
        model = ResNet101()
        target_layers = [model.relu]
    elif mdl_name == 'ResNet50_MLP_2':
        model = ResNet50_MLP_2()
    elif mdl_name == 'ResNet50_MLP_3':
        model = ResNet50_MLP_3()
    elif mdl_name == 'ResNet50_BN':
        model = ResNet50_BN()
    else:
        model = None

    model.to(device).eval()
    model.load_state_dict(torch.load(ckpt_path, map_location=torch.device('cpu')))
    pytorch_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('param num:', pytorch_total_params)
    print(f'Ground truth {gt}')

    if cam_type is not None:
        print('>>>>>> Saliency Map Generating...')
    else:
        print('>>>>>> Saliency Map Generation Disabled.')

    if grayscale:
        print('>>>>>> Grayscale Enabled.')
    else:
        print('>>>>>> Grayscale Disabled.')

    # save the counts of each combination for the calculation of sensitivity and specificity
    stat_list = [0, 0, 0, 0, 0]  # [Kras_7d, Kras_21d, Kras_9w, Bcat_7d, Bcat_9w]
    for img in tqdm(imgs):
        if img == '.DS_Store' or img == 'error' or 'mask' in img:
            continue
        img_path = os.path.join(testset, img)

        if cam_type is not None:
            cls_target = args.cls_target
            test_info = testset.split('/')[-1].split('_')
            mouse_id = test_info[0]
            gof = test_info[1]
            lof = test_info[2]
            time = test_info[3]
            stain = test_info[4]  # not ready
            if cls_target == 0:
                cls_type = 'm_kras'
            elif cls_target == 1:
                cls_type = 'm_bcat'
            elif cls_target == 2:
                cls_type = 't_7d'
            elif cls_target == 3:
                cls_type = 't_21d'
            else:
                cls_type = 't_9w'

            out_path = os.path.join(os.path.join('saliency', mdl_name), cam_type + '_' + mouse_id + '_' + gof + '_' + lof + '_' + time + '-' + cls_type)

            # check whether the region (i.e., [x=1234, y=1234]) is indicated in the file name
            # for the current case, no region coordinate in file name
            if len(img.split(' ')) <= 1:
                region = img
            else:
                region = img.split(' ')[1]

            # get heatmap
            heatmap, heatmap_with_img, res = utils.get_saliency_map(model, target_layers, cls_target, img_path, cam_type, grayscale)

            # Check whether folder exists
            utils.check_folder_existence(out_path)
            utils.check_folder_existence(out_path + '/heat')

            # Use PIL.save to ensure saving images in RGB format (DO NOT USE cv2!!!!)
            out_heat = Image.fromarray(heatmap)
            out_im = Image.fromarray(heatmap_with_img)
            out_heat.save(os.path.join(out_path, 'heat/' + res + '_' + region))
            out_im.save(os.path.join(out_path, res + '_' + region))
        else:
            img = utils.load_img(img_path, grayscale, True)
            with torch.no_grad():
                pred_gene, pred_time = model(img)
            pred = utils.res_in_binary(torch.softmax(pred_gene, dim=1))[0] + utils.res_in_binary(torch.softmax(pred_time, dim=1))[0]
            res = utils.pred_to_text(pred)

        # check correct cases
        if res == gt:
            correct_num += 1

        if eval_as_fold:
            if res == 'Kras_P53_7d':
                stat_list[0] += 1
            elif res == 'Kras_P53_21d':
                stat_list[1] += 1
            elif res == 'Kras_P53_9w':
                stat_list[2] += 1
            elif res == 'Bcat_P53_7d':
                stat_list[3] += 1
            elif res == 'Bcat_P53_9w':
                stat_list[4] += 1

    # we use total_num / 2 here because there are original image and mask in one folder
    stdout.write(f"{correct_num} out of {int(total_num / 2)} are correct. The eval accuracy is {correct_num / (total_num / 2): .4f}.\n")
    return args.testset.split('/')[-1], correct_num, total_num, stat_list


if __name__ == '__main__':
    args = get_args()
    if args.eval_as_fold:
        testset, correct_num, total_num, stat_list = evaluation(args)
        acc = correct_num / ((total_num - 1) / 2)
        if os.path.exists('res.csv'):
            with open('res.csv', 'a', encoding='UTF-8') as f:
                writer = csv.writer(f)
                writer.writerow([testset, int((total_num - 1) / 2), correct_num, acc, stat_list[0], stat_list[1], stat_list[2], stat_list[3], stat_list[4]])
        else:
            with open('res.csv', 'w', encoding='UTF-8') as f:
                writer = csv.writer(f)
                writer.writerow(['testset', 'total_num', 'correct_num', 'acc', 'Kras_7d', 'Kras_21d', 'Kras_9w', 'Bcat_7d', 'Bcat_9w'])
                writer.writerow([testset, int((total_num - 1) / 2), correct_num, acc, stat_list[0], stat_list[1], stat_list[2], stat_list[3], stat_list[4]])
                f.close()
    else:
        evaluation(args)
