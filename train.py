from utils import *
import torch.utils.data
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.transforms.functional as tf
from dataset import *
from all_models import *
import torch.nn.functional as F

from sys import stdout
import os
import argparse
import random


# Train with GPU
# The mps is for M1 chip Mac only. If not working, comment it and use the code in line 24.
# device = torch.device("cuda:0" if torch.cuda.is_available() else ("mps" if torch.has_mps else "cpu"))
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def compute_loss(out_gene, out_time, gene_label, time_label, loss_ratio):
    """
    The implementation of loss function
    out_gene/out_time: model prediction
    gene_label/time_label: ground truth
    loss_ratio: in this project, 0.3 is the best
    """
    loss = loss_ratio * F.cross_entropy(out_gene, gene_label) + \
           (1 - loss_ratio) * F.cross_entropy(out_time, time_label)
    return loss


def get_args():
    parser = argparse.ArgumentParser('train')

    # Add model architecture
    parser.add_argument('--arch', default='ResNet50', help='Specify one model architecture')

    # Add dataset args
    parser.add_argument('--data', default='data', help='Specify data folder')
    parser.add_argument('--annotations', default='annotation.csv', help='Specify annotation file')
    parser.add_argument('--train-percent', default=0.6, type=int)

    # Add training params
    parser.add_argument('--epoch', default=50, type=int)
    parser.add_argument('--batch-size', default=64, type=int)
    parser.add_argument('--lr', default=0.01, type=float)
    parser.add_argument('--grayscale', default=False, type=bool, action=argparse.BooleanOptionalAction, help='Enable grayscale')
    parser.add_argument('--loss-ratio', default=0.3, type=float, help='Ratio of gene and time in combine loss computation')

    # Add logging param
    parser.add_argument('--interval', default=10, type=int, help='Interval of iteration for printing log in each epoch')

    args = parser.parse_args()
    return args


def validation(model, epoch, val_loader, interval, loss_ratio):
    model.eval()
    gene_total_correct = 0
    time_total_correct = 0
    gene_interval_acc = []
    time_interval_acc = []
    interval_loss = []
    overall_acc = 0
    sample_num = len(val_loader)
    stdout.write(f"-------- Starting validating epoch {epoch + 1} --------\n")
    for idx, (data, gene_targets, time_targets) in enumerate(val_loader):
        # Load validation data
        data = data.to(device, dtype=torch.float32)  # image
        gene_targets = gene_targets.to(device, dtype=torch.float32)  # mutation label
        time_targets = time_targets.to(device, dtype=torch.float32)  # time label
        model.zero_grad()

        # Use no_grad because the validation set is not used for development
        with torch.no_grad():
            gene, time = model(data)
            loss = compute_loss(gene, time, gene_targets, time_targets, loss_ratio)

        # Transform output and labels to a list for accuracy computation
        gene_preds = res_in_binary(torch.softmax(gene, dim=1))  # list of [0, 1]/[1, 0]
        time_preds = res_in_binary(torch.softmax(time, dim=1))  # list of [1, 0, 0]/[0, 1, 0]/[0, 0, 1]
        gene_labels = res_in_binary(gene_targets)
        time_labels = res_in_binary(time_targets)

        # Compute metrics e.g. accuracy and loss for this interval
        gene_interval_correct = sum([1 for i in range(len(gene_preds)) if gene_preds[i] == gene_labels[i]])
        time_interval_correct = sum([1 for i in range(len(time_preds)) if time_preds[i] == time_labels[i]])
        gene_interval_acc.append(gene_interval_correct / len(gene_preds))
        time_interval_acc.append(time_interval_correct / len(time_preds))
        interval_loss.append(loss.item())

        # Store accuracy information
        gene_total_correct += gene_interval_correct / len(gene_preds)
        time_total_correct += time_interval_correct / len(time_preds)

        # overall accuracy is also computed based on the loss ratio set in the loss function
        overall_acc = loss_ratio * (gene_total_correct / sample_num) + (1 - loss_ratio) * (time_total_correct / sample_num)

        # Print logging information
        if (idx + 1) % interval == 0:
            stdout.write(f"Epoch({epoch + 1}) [{idx + 1}/{len(val_loader)}] - Loss: {sum(interval_loss) / len(interval_loss):.5f}, Gene Acc.: {sum(gene_interval_acc) / len(gene_interval_acc):.3f}, Time Acc.: {sum(time_interval_acc) / len(time_interval_acc):.3f}\n")
            interval_loss = []
        elif (idx + 1) == sample_num:
            stdout.write(f"Epoch({epoch + 1}) [{idx + 1}/{len(val_loader)}] - Loss: {sum(interval_loss) / len(interval_loss):.5f}, Gene Acc.: {sum(gene_interval_acc) / len(gene_interval_acc):.3f}, Time Acc.: {sum(time_interval_acc) / len(time_interval_acc):.3f}\n")
    # Print epoch validation acc.
    stdout.write(f"Epoch({epoch + 1}) - Overall Validation Accuracy: {overall_acc:.3f}\n")
    # Return the validation acc. for checkpoint selection
    return overall_acc


def train(args):
    # training params
    num_epochs = args.epoch
    batch_size = args.batch_size
    lr = args.lr
    grayscale = args.grayscale
    loss_ratio = args.loss_ratio
    data = args.data
    annotations = args.annotations

    # prepare dataset
    dataset = GeneAndTime(annotations=annotations, img_dir=data, grayscale=grayscale)
    data_volume = len(dataset)
    train_num = int(args.train_percent * data_volume)
    val_num = data_volume - train_num

    # randomly split data according to train and val ratio
    train_set, val_set = torch.utils.data.random_split(dataset, [train_num, val_num])
    train_loader = DataLoader(dataset=train_set, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(dataset=val_set, batch_size=batch_size, shuffle=True)

    # load model
    mdl_name = args.arch
    model = eval(mdl_name + '()').to(device)
    print(model)

    # optimizer and lr scheduler
    optimizer = optim.SGD(model.parameters(), momentum=0.9, lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=18, gamma=0.1)

    # Save model state
    # This path should be further modified if it is not run on a GoogleColab Notebook
    if torch.cuda.is_available():
        ckpt_dir = os.path.join('drive/MyDrive/hcc_detection/ckpt', mdl_name)
    else:
        ckpt_dir = os.path.join('ckpt', mdl_name)

    if grayscale:
        print('>>>>>> Grayscale Enabled.')
    else:
        print('>>>>>> Grayscale Disabled.')

    # initialise variables
    best_epoch = 0
    best_overall = 0
    best_val_epoch = 0
    best_val = 0
    interval = args.interval
    for epoch in range(num_epochs):
        # set model to train
        model.train()
        gene_total_correct = 0
        time_total_correct = 0
        gene_interval_acc = []
        time_interval_acc = []
        interval_loss = []
        stdout.write(f"-------- Starting training epoch {epoch + 1} --------\n")
        sample_num = len(train_loader)
        for idx, (data, gene_targets, time_targets) in enumerate(train_loader):
            angle = random.choice([0, 90, 180, 270])  # set four angles for rotation
            data = data.to(device, dtype=torch.float32)  # image
            data = tf.rotate(data, angle)  # apply random rotation
            gene_targets = gene_targets.to(device, dtype=torch.float32)  # mutation label
            time_targets = time_targets.to(device, dtype=torch.float32)  # time label
            model.zero_grad()

            gene, time = model(data)  # get model prediction
            loss = compute_loss(gene, time, gene_targets, time_targets, loss_ratio)  # compute loss
            loss.backward()
            optimizer.step()

            # Transform output and labels to a list for accuracy computation
            gene_preds = res_in_binary(torch.softmax(gene, dim=1))  # list of [0, 1]/[1, 0]
            time_preds = res_in_binary(torch.softmax(time, dim=1))  # list of [1, 0, 0]/[0, 1, 0]/[0, 0, 1]
            gene_labels = res_in_binary(gene_targets)
            time_labels = res_in_binary(time_targets)

            # Compute metrics e.g. accuracy and loss for this interval
            gene_interval_correct = sum([1 for i in range(len(gene_preds)) if gene_preds[i] == gene_labels[i]])
            time_interval_correct = sum([1 for i in range(len(time_preds)) if time_preds[i] == time_labels[i]])
            gene_interval_acc.append(gene_interval_correct / len(gene_preds))
            time_interval_acc.append(time_interval_correct / len(time_preds))
            interval_loss.append(loss.item())

            # Store accuracy information
            gene_total_correct += gene_interval_correct / len(gene_preds)
            time_total_correct += time_interval_correct / len(time_preds)

            # Print log for interval
            if (idx + 1) % interval == 0:
                stdout.write(
                    f"Epoch({epoch + 1}) [{idx + 1}/{len(train_loader)}] - Loss: {sum(interval_loss) / len(interval_loss):.5f}, Gene Acc.: {sum(gene_interval_acc) / len(gene_interval_acc):.3f}, Time Acc.: {sum(time_interval_acc) / len(time_interval_acc):.3f}\n")
                gene_interval_acc = []
                time_interval_acc = []
                interval_loss = []
            elif (idx + 1) == sample_num:
                stdout.write(
                    f"Epoch({epoch + 1}) [{idx + 1}/{len(train_loader)}] - Loss: {sum(interval_loss) / len(interval_loss):.5f}, Gene Acc.: {sum(gene_interval_acc) / len(gene_interval_acc):.3f}, Time Acc.: {sum(time_interval_acc) / len(time_interval_acc):.3f}\n")
        scheduler.step()

        # Overall accuracy is also computed based on the loss ratio set in the loss function
        overall_acc = loss_ratio * (gene_total_correct / sample_num) + (1 - loss_ratio) * (time_total_correct / sample_num)

        # Find best acc.
        if overall_acc >= best_overall:
            best_epoch = epoch + 1
            best_overall = overall_acc

        # Print log for epoch
        stdout.write(f"Epoch({epoch + 1}) - Overall Training Accuracy: {overall_acc:.3f}\n")
        stdout.write(f"The best accuracy overall is: Epoch {best_epoch} with {best_overall:.3f}\n")

        # Do validation
        val_acc = validation(model, epoch, val_loader, interval, loss_ratio)

        # Find best val acc.
        if val_acc >= best_val:
            best_val_epoch = epoch + 1
            best_val = val_acc
        stdout.write(f"The best validation accuracy is: Epoch {best_val_epoch} with {best_val:.3f}\n")

        # Save model
        save_model_state(model, ckpt_dir, epoch + 1, best_val_epoch)


if __name__ == '__main__':
    args = get_args()
    train(args)

