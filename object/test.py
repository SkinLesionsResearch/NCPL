import argparse
import os, sys
os.chdir("/home/jackie/ResearchArea/SkinCancerResearch/semi_skin_cancer")
sys.path.append("/home/jackie/ResearchArea/SkinCancerResearch/semi_skin_cancer")
print(os.getcwd())
import os.path as osp
import torchvision
import numpy as np
import torch
# import torch.nn as nn
# import torch.optim as optim
# from itertools import cycle
from torchvision import transforms
# import network, loss
from torch.utils.data import DataLoader
from data_list import ImageList, ImageList_idx
import random, pdb, math, copy
from evaluation.draw import draw_ROC, draw_TSNE, draw_cm
from evaluation.metrics import get_metrics, get_test_data
import matplotlib.pyplot as plt
from transforms import image_test
import utils

plt.rc('font', family='Times New Roman')

def op_copy(optimizer):
    for param_group in optimizer.param_groups:
        param_group['lr0'] = param_group['lr']
    return optimizer

def image_test(resize_size=256, crop_size=224, alexnet=False):
    if not alexnet:
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])
    else:
        normalize = Normalize(meanfile='./ilsvrc_2012_mean.npy')
    return transforms.Compose([
        transforms.Resize((resize_size, resize_size)),
        # transforms.CenterCrop(crop_size),
        transforms.ToTensor(),
        normalize
    ])


def data_load(args):
    ## prepare data
    dsets = {}
    dset_loaders = {}
    test_txt = open(osp.join(args.dset_path, 'test.txt')).readlines()
    dsets["test"] = ImageList(test_txt, args, transform=image_test())
    dset_loaders["test"] = DataLoader(dsets["test"], batch_size=args.batch_size, shuffle=True,
                                      num_workers=args.worker, drop_last=False)

    return dset_loaders

def print_args(args):
    s = "==========================================\n"
    for arg, content in args.__dict__.items():
        s += "{}:{}\n".format(arg, content)
    return s

def draw_heatmap(model, img_dir, save_dir, transform=None, visual_heatmap=False):
    for file in os.listdir(img_dir):
        if os.path.isfile(os.path.join(img_dir, file)):
            pass
            # draw_CAM(model, os.path.join(img_dir, file), save_dir, transform, visual_heatmap)


def test_target(args):
    SEED = args.seed
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
    np.random.seed(SEED)
    random.seed(SEED)

    ## set base network
    net = utils.get_model(args.net, args.num_classes)

    args.modelpath = args.output_dir_train + '/best_params.pt'
    print(args.modelpath)
    net.load_state_dict(torch.load(args.modelpath))

    net.eval()

    if not args.draw_cam:
        print("run not draw_cam")
        dset_loaders = data_load(args)
        features, logits, y_true, y_predict = get_test_data(dset_loaders['test'], net)
        accuracy, kappa, report, sensitivity, specificity, roc_auc = get_metrics(logits, y_true, y_predict)
        draw_ROC(logits, y_true, args.label_names, args.output_dir)
        draw_cm(y_true, y_predict, args.label_names, args.output_dir)
        draw_TSNE(features, y_true, args.label_names, args.output_dir)

        log_str = '\nAccuracy = {:.2f}%, Kappa = {:.4f},' \
                  ' Sensitivity = {:.4f}, Specificity = {:.4f}, AUROC = {:.4f}\n' \
            .format(accuracy, kappa, sensitivity, specificity, roc_auc)

        args.out_file.write(log_str)
        args.out_file.write(report)
        args.out_file.flush()
        print(log_str)
        print(report)
    else:
        print("run draw_cam")
        draw_heatmap(net, args.img_dir, args.save_dir, image_test(), True)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='oral_cancer')
    parser.add_argument('--gpu_id', type=str, nargs='?', default='0', help="device id to run")
    parser.add_argument('--batch_size', type=int, default=32, help="batch_size")
    parser.add_argument('--num_classes', type=int, default=7, help="number of classes")
    parser.add_argument('--worker', type=int, default=12, help="number of workers")
    parser.add_argument('--dir', type=str, default='./ckps/')
    parser.add_argument('--subDir', type=str, default='resnet50_sev_cates_2500_0.99_naive_0_afm_0.7_u_0.3')
    parser.add_argument('--dset_path', type=str, default='./data/semi_processed')
    parser.add_argument('--seed', type=int, default=2021, help="random seed")
    parser.add_argument('--which', type=str, default='one', choices=['one', 'all'])
    parser.add_argument('--draw_cam', type=bool, default=False)
    parser.add_argument('--img_dir', type=str, default=None)
    parser.add_argument('--save_dir', type=str, default=None)

    args = parser.parse_args()
    args.label_names = ['akiec', 'bcc', 'bkl', 'df', 'mel', 'nv', 'vasc']

    if args.which == 'one':
        args.net = osp.basename(args.subDir).split('_')[0]
        # torch.backends.cudnn.deterministic = True
        print(args.dir)

        args.output_dir_train = os.path.join(args.dir, args.subDir)
        print(args.output_dir_train)
        args.output_dir = os.path.join('test', args.output_dir_train)

        if not osp.exists(args.output_dir):
            os.system('mkdir -p ' + args.output_dir)
        if not osp.exists(args.output_dir):
            os.makedirs(args.output_dir)

        args.out_file = open(osp.join(args.output_dir, 'log.txt'), 'w')
        args.out_file.write(print_args(args) + '\n')
        args.out_file.flush()

        test_target(args)

    if args.which == 'all':
        for dir in os.listdir(args.dir):
            args.net = dir.split('_')[0]
            # torch.backends.cudnn.deterministic = True

            args.output_dir_train = os.path.join(args.dir, dir)
            args.output_dir = os.path.join('./test', args.output_dir_train)

            if not osp.exists(args.output_dir):
                os.system('mkdir -p ' + args.output_dir)
            if not osp.exists(args.output_dir):
                os.makedirs(args.output_dir)

            args.out_file = open(osp.join(args.output_dir, 'log.txt'), 'w')
            args.out_file.write(print_args(args) + '\n')
            args.out_file.flush()

            test_target(args)
