import os
import sys

os.chdir("/home/jackie/ResearchArea/SkinCancerResearch/semi_skin_cancer")
sys.path.append("/home/jackie/ResearchArea/SkinCancerResearch/semi_skin_cancer")
print(os.getcwd())
import argparse
from data_list import ImageListWithPath
from torch.utils.data import DataLoader
import utils

import cv2
import torch
import torch.nn.functional as F
import numpy as np
import torchvision.transforms as transforms
from PIL import ImageFile

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


def get_data_loader(args):
    ## prepare data
    test_txt = open(args.test_img_dir).readlines()
    dsets = ImageListWithPath(test_txt, args, transform=image_test())
    dset_loader = DataLoader(dsets, batch_size=args.batch_size, shuffle=True,
                             num_workers=args.worker, drop_last=False)

    return dset_loader


def gen_cam(img_path, logits, net, idx_imgs, suffix):
    # return
    h_x = F.softmax(logits, dim=0).data.squeeze()
    probs, idx = torch.sort(h_x)
    idx = idx.numpy()

    model_state_dict = net.state_dict()

    fc_weights = model_state_dict["classifier.weight"].cpu().numpy()  # [2,2048]
    features_conv = net.ori_features_ly4.data[idx_imgs].cpu().numpy()
    features_conv = np.array([features_conv])

    CAMs = return_CAM(features_conv, fc_weights, [idx[0]])

    img_ori = cv2.imread(img_path)
    height, width, _ = img_ori.shape
    heatmap = cv2.applyColorMap(cv2.resize(CAMs[0], (width, height)), cv2.COLORMAP_JET)
    img_cam = heatmap * 0.5 + img_ori * 0.5
    img_name = img_path.split("/")[-1].split(".")[0]
    cv2.imwrite(os.path.join(args.save_dir, (img_name + ".jpg")), img_ori)
    cv2.imwrite(os.path.join(args.save_dir, (img_name + "_" + suffix + "_cam.jpg")), img_cam)


def target_logic(args, suffix):
    test_loader = get_data_loader(args)

    net = utils.get_model(args.net, args.num_classes)
    print(args.model_path)

    is_model_state_dict_valid = net.load_state_dict(torch.load(args.model_path))
    print(is_model_state_dict_valid)
    net.eval()

    _, logits, _, _, all_path = get_test_data(test_loader, net)
    for i in range(len(logits)):
        gen_cam(all_path[i], logits[i], net, i, suffix)


def gen_path():
    args.net = os.path.basename(args.sub_dir).split('_')[0]
    args.output_dir_train = ""
    args.output_dir_train = os.path.join(args.output_dir_train, args.dir)
    args.output_dir_train = os.path.join(args.output_dir_train, args.sub_dir)
    args.model_path = args.output_dir_train + '/best_params.pt'


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='oral_cancer')
    parser.add_argument('--batch_size', type=int, default=1, help="batch_size")
    parser.add_argument('--num_classes', type=int, default=2, help="number of classes")
    parser.add_argument('--worker', type=int, default=12, help="number of workers")
    parser.add_argument('--dir', type=str, default='./ckps/')
    parser.add_argument('--sub_dir', type=str, default='resnet50_tc_bcc_2500_0.99_naive_0_afm_0.7_u_0.3')
    parser.add_argument('--test_img_dir', type=str, default='./data/semi_processed_bcc/test_cam.txt')
    parser.add_argument('--dset_path', type=str, default='./data/semi_processed_bcc')
    parser.add_argument('--save_dir', type=str, default="cam_res")

    args = parser.parse_args()

    if not os.path.exists(args.save_dir):
        os.mkdir(args.save_dir)

    args.dset_path = './data/semi_processed_bcc'
    args.batch_size = 500

    # gen imgs for the ncpl
    args.dir = "ckps"
    args.sub_dir = "resnet50_tc_bcc_2500_0.99_naive_0_afm_0.7_u_0.3"

    gen_path()
    target_logic(args, "ncpl")

    # gen imgs for the baseline resnet50
    args.dir = "ckps_bl"
    args.sub_dir = "resnet50_tc_bcc_2500_0.0_naive_0_afm_0.7_u_0.3"

    gen_path()
    target_logic(args, "bl")
