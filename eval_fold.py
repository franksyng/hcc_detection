import os
import argparse
from eval import evaluation


def get_args():
    parser = argparse.ArgumentParser('eval_fold')

    parser.add_argument('--arch', default='ResNet50', help='Specify one model architecture')
    parser.add_argument('--ckpt', default='ckpt/ResNet50/best.pth', help='Choose checkpoint file')
    parser.add_argument('--fold-path', help='Path to the fold')
    parser.add_argument('--cls-type', default='mutation', type=str, help='The desired type of cls category (mutation/time)')
    parser.add_argument('--saliency', default=None, type=str, help='Enable saliency map')
    parser.add_argument('--grayscale', default=False, type=bool, action=argparse.BooleanOptionalAction, help='Enable grayscale')
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = get_args()

    fold_path = args.fold_path
    dirs = os.listdir(fold_path)
    ckpt = args.ckpt
    model = args.arch
    cls_type = args.cls_type
    grayscale = args.grayscale
    cam_type = args.saliency

    for each_dir in dirs:
        if each_dir != '.DS_Store':
            mouse_path = os.path.join(fold_path, each_dir)
            images = os.listdir(mouse_path)
            info = each_dir.split('/')[-1].split('_')
            gof = info[1]
            lof = info[2]
            time = info[3]
            gt = gof + '_' + lof + '_' + time
            if cam_type is not None:
                if cls_type == 'mutation':
                    if gof == 'Kras':
                        cls_target = str(0)
                    else:
                        cls_target = str(1)
                    cmd_str = ' --cls-target ' + cls_target + ' --saliency ' + cam_type
                else:
                    if time == '7d':
                        cls_target = str(2)
                    elif time == '21d':
                        cls_target = str(3)
                    else:
                        cls_target = str(4)
                    cmd_str = ' --cls-target ' + cls_target + ' --saliency ' + cam_type
            else:
                cmd_str = ''
            if grayscale:
                cmd = 'python eval.py --arch ' + model + ' ' \
                      '--testset ' + os.path.join(fold_path, each_dir) + ' ' \
                      '--ckpt ' + ckpt + ' --gt ' + gt + ' ' \
                      '--grayscale --eval-as-fold' + cmd_str
            else:
                cmd = 'python eval.py --arch ' + model + ' ' \
                      '--testset ' + os.path.join(fold_path, each_dir) + ' ' \
                      '--ckpt ' + ckpt + ' --gt ' + gt + ' ' \
                      '--eval-as-fold' + cmd_str
            os.system(cmd)
