import os
import sys

os.chdir("/home/jackie/ResearchArea/SkinCancerResearch/semi_skin_cancer")
sys.path.append("/home/jackie/ResearchArea/SkinCancerResearch/semi_skin_cancer")
print(os.getcwd())
import argparse
import torch
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from PIL import ImageFile, Image
from data_list import ImageList, ImageListWithPath
from torch.utils.data import DataLoader
import utils
import numpy as np
import cv2

import cv2
import torch
import torch.nn as nn
from torchvision import models
import torch.nn.functional as F
import numpy as np
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.autograd import Variable
from PIL import Image
from PIL import ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True

ImageFile.LOAD_TRUNCATED_IMAGES = True


def get_test_data(loader, net):
    start_test = True
    with torch.no_grad():
        iter_test = iter(loader)
        for i in range(len(loader)):
            data = iter_test.next()
            inputs = data[0]
            labels = data[1]
            path = data[2]
            inputs = inputs.cuda()
            features, outputs = net(inputs)
            if start_test:
                all_features = features.float().cpu()
                all_output = outputs.float().cpu()
                all_label = labels.float()
                all_path = path
                start_test = False
            else:
                all_features = torch.cat((all_features, features.float().cpu()), 0)
                all_output = torch.cat((all_output, outputs.float().cpu()), 0)
                all_label = torch.cat((all_label, labels.float()), 0)
                all_path += path
    _, predict = torch.max(all_output, 1)
    return all_features, all_output, all_label, predict, all_path


def return_CAM(feature_conv, weight, class_idx):
    # generate the class -activation maps upsample to 256x256
    size_upsample = (256, 256)
    bz, nc, h, w = feature_conv.shape

    output_cam = []
    for idx in class_idx:
        feature_conv = feature_conv.reshape((nc, h * w))
        cam = np.matmul(weight[idx], feature_conv)
        cam = cam.reshape(h, w)
        cam = cam - np.min(cam)
        cam_img = cam / np.max(cam)
        cam_img = np.uint8(255 * cam_img)
        output_cam.append(cv2.resize(cam_img, size_upsample))
    return output_cam


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


def get_data_loaders(args, test_fname):
    ## prepare data
    dsets = {}
    dset_loaders = {}
    test_txt = open(os.path.join(args.dset_path, test_fname)).readlines()
    dsets[test_fname] = ImageListWithPath(test_txt, args, transform=image_test())
    dset_loaders[test_fname] = DataLoader(dsets[test_fname], batch_size=args.batch_size, shuffle=True,
                                          num_workers=args.worker, drop_last=False)

    return dset_loaders


def target_logic(args, suffix):
    args.batch_size = 32
    test_loader = get_data_loaders(args, "test_cam_1.txt")["test_cam_1.txt"]

    net = utils.get_model(args.net, args.num_classes)
    print(args.model_path)

    is_model_state_dict_valid = net.load_state_dict(torch.load(args.model_path))
    print(is_model_state_dict_valid)
    net.eval()

    _, logits, _, _, all_path = get_test_data(test_loader, net)

    img_path = all_path[0]
    logits = logits[0]
    h_x = F.softmax(logits, dim=0).data.squeeze()
    probs, idx = torch.sort(h_x)
    probs = probs.detach().numpy()
    idx = idx.numpy()

    model_state_dict = net.state_dict()
    # for k, v in model_state_dict:
    #     print(k, ", ", v)

    fc_weights = model_state_dict["classifier.weight"].cpu().numpy()  # [2,2048]
    features_conv = net.ori_features_ly4.cpu().numpy()
    CAMs = return_CAM(features_conv, fc_weights, [idx[0]])
    flist_path = os.path.join(args.dset_path, "test_cam.txt")
    test_txt = open(flist_path).readlines()

    img_ori = cv2.imread(img_path)
    height, width, _ = img_ori.shape
    heatmap = cv2.applyColorMap(cv2.resize(CAMs[0], (width, height)), cv2.COLORMAP_JET)
    img_cam = heatmap * 0.5 + img_ori * 0.5
    img_name = img_path.split("/")[-1].split(".")[0]
    cv2.imwrite(img_name + ".jpg", img_ori)
    cv2.imwrite(img_name + "_" + suffix + "_cam.jpg", img_cam)


def gen_path():
    args.net = os.path.basename(args.subDir).split('_')[0]
    args.output_dir_train = ""
    args.output_dir_train = os.path.join(args.output_dir_train, args.dir)
    args.output_dir_train = os.path.join(args.output_dir_train, args.subDir)
    args.model_path = args.output_dir_train + '/best_params.pt'


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='oral_cancer')
    parser.add_argument('--gpu_id', type=str, nargs='?', default='0', help="device id to run")
    parser.add_argument('--batch_size', type=int, default=32, help="batch_size")
    parser.add_argument('--num_classes', type=int, default=2, help="number of classes")
    parser.add_argument('--worker', type=int, default=12, help="number of workers")
    parser.add_argument('--dir', type=str, default='./ckps/')
    parser.add_argument('--subDir', type=str, default='resnet50_tc_bcc_2500_0.99_naive_0_afm_0.7_u_0.3')
    parser.add_argument('--output_dir_train', type=str,
                        default='/home/jackie/ResearchArea/SkinCancerResearch/semi_skin_cancer')
    parser.add_argument('--dset_path', type=str, default='./data/semi_processed_bcc')
    parser.add_argument('--seed', type=int, default=2021, help="random seed")
    parser.add_argument('--which', type=str, default='one', choices=['one', 'all'])
    parser.add_argument('--draw_cam', type=bool, default=False)
    parser.add_argument('--img_dir', type=str, default=None)
    parser.add_argument('--save_dir', type=str, default=None)

    args = parser.parse_args()

    args.dset_path = './data/semi_processed_bcc'

    # gen imgs for the ncpl
    args.dir = "ckps"
    args.subDir = "resnet50_tc_bcc_2500_0.99_naive_0_afm_0.7_u_0.3"

    gen_path()
    target_logic(args, "ncpl")

    # gen imgs for the baseline resnet50
    args.dir = "ckps_bl"
    args.subDir = "resnet50_tc_bcc_2500_0.0_naive_0_afm_0.7_u_0.3"

    gen_path()
    target_logic(args, "bl")
