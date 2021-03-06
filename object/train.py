import argparse
import os, sys

sys.path.append("..")
sys.path.append(".")
import os.path as osp
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
from torch.utils.data import DataLoader
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from data_list import ImageList, ImageList_idx, ImageList_confident
import json
import random
from evaluation.metrics import get_metrics, get_metrics_sev_class, get_test_data
from object.transforms import image_test, image_train
from object.imbalanced import ImbalancedDatasetSampler
from object import utils
from loss import UncertaintyLoss, UncertaintyLoss, FocalLossClassProportion

import warnings

warnings.filterwarnings("ignore")


def op_copy(optimizer):
    for param_group in optimizer.param_groups:
        param_group['lr0'] = param_group['lr']
    return optimizer


def data_load(args):
    ## prepare data
    dsets = {}
    dset_loaders = {}
    train_x_txt = open(osp.join(args.src_dset_path, 'train', str(args.labeled_num), 'train_labeled.txt')).readlines()
    train_u_txt = open(osp.join(args.src_dset_path, 'train', str(args.labeled_num), 'train_unlabeled.txt')).readlines()
    test_txt = open(osp.join(args.src_dset_path, 'test.txt')).readlines()
    if args.net[0:5] == "senet":
        image_train_transform = image_train(299)
        image_test_transform = image_test(299)
    elif args.net[0:3] == "ran":
        image_train_transform = image_train(32)
        image_test_transform = image_test(32)
    else:
        image_train_transform = image_train()
        image_test_transform = image_test()

    dsets["train_x"] = ImageList(train_x_txt, args, transform=image_train_transform)
    dsets["train_u"] = ImageList_idx(train_u_txt, args, transform=image_train_transform)
    if args.imb:
        dset_loaders["train_x"] = DataLoader(dsets["train_x"], batch_size=args.batch_size,
                                             sampler=ImbalancedDatasetSampler(dsets["train"]),
                                             num_workers=args.worker, drop_last=True)
    else:
        dset_loaders["train_x"] = DataLoader(dsets["train_x"], batch_size=args.batch_size, shuffle=True,
                                             num_workers=args.worker, drop_last=True)

    dset_loaders["train_u"] = DataLoader(dsets["train_u"], batch_size=args.batch_size, shuffle=False,
                                         num_workers=args.worker, drop_last=False)

    dsets["test"] = ImageList(test_txt, args, transform=image_test_transform)
    dset_loaders["test"] = DataLoader(dsets["test"], batch_size=args.batch_size, shuffle=True,
                                      num_workers=args.worker, drop_last=True)
    print('Labeled training Data Count:', len(dsets["train_x"]), ', Distribution:')
    df_train_labeled = pd.DataFrame(dsets["train_x"].imgs, columns=['path', 'class'])
    print(df_train_labeled.groupby('class').count())

    print('Unlabeled training Data Count:', len(dsets["train_u"]), ', Distribution:')
    df_train_unlabeled = pd.DataFrame(dsets["train_u"].imgs, columns=['path', 'class'])
    print(df_train_unlabeled.groupby('class').count())


    args.class_proportion = np.array((df_train_labeled.groupby('class').count() + \
                                    df_train_unlabeled.groupby('class').count())['path'].tolist())
    args.class_proportion = 1 / args.class_proportion
    # args.class_proportion = args.class_proportion / sum(args.class_proportion)
    print("class_proportion for training data")
    print(args.class_proportion)

    print('\nTest Data Count:', len(dsets["test"]), ', Distribution:')
    df_test = pd.DataFrame(dsets["test"].imgs, columns=['path', 'class'])
    print(df_test.groupby('class').count())
    args.class_proportion_test = np.array(df_test.groupby('class')['path'].count().tolist())
    args.class_proportion_test = 1 / args.class_proportion_test
    print("class_proportion for test data")
    print(args.class_proportion_test)

    return dset_loaders


# get the samples whose confident level(probs predicted by the initial step model)
# larger than a threshold
def obtain_confident_loader(loader, net, args):
    start_test = True

    if args.net[0:5] == "senet":
        image_train_transform = image_train(299)
    elif args.net[0:3] == "ran":
        image_train_transform = image_train(32)
    else:
        image_train_transform = image_train()

    with torch.no_grad():
        iter_test = iter(loader)
        for _ in range(len(loader)):
            data = iter_test.next()
            inputs = data[0]
            labels = data[1]
            indices = data[2]
            inputs = inputs.cuda()
            features, logits = net(inputs)
            if start_test:
                all_fea = features.float().cpu()
                all_output = logits.float().cpu()
                all_label = labels.float()
                all_index = indices.int().cpu()
                start_test = False
            else:
                all_fea = torch.cat((all_fea, features.float().cpu()), 0)
                all_output = torch.cat((all_output, logits.float().cpu()), 0)
                all_label = torch.cat((all_label, labels.float()), 0)
                all_index = torch.cat((all_index, indices), 0)

    all_output = nn.Softmax(dim=1)(all_output)
    # torch.max(_, 1): get the max number and corresponding indices for each line(dim==1)
    prob, predict = torch.max(all_output, 1)
    confident_indices = all_index[prob > args.threshold]

    all_accuracy = torch.sum(
        torch.squeeze(predict).float() == all_label).item() / float(all_label.size()[0])
    confident_accuracy = torch.sum(
        torch.squeeze(
            predict[confident_indices]).float() == all_label[confident_indices]).item() / float(len(confident_indices))
    log_str = 'All Accuracy = {:.2f}%, Confident Accuracy = {:.2f}%, Confident Data Count: {}' \
        .format(all_accuracy * 100, confident_accuracy * 100, len(confident_indices))

    args.out_file.write(log_str + '\n')
    args.out_file.flush()
    print(log_str + '\n')

    # Create Confident Dataloader
    train_u_txt = open(
        osp.join(args.src_dset_path, 'train', str(args.labeled_num), 'train_unlabeled.txt')).readlines()
    # get the samples which is confident samples to make labels
    dset = ImageList_confident(
        [train_u_txt[i] for i in confident_indices], args, pseudo_labels=predict[confident_indices].squeeze().numpy(), \
        real_labels=all_label[confident_indices].int().numpy(), transform=image_train_transform)
    confident_dset_loader = DataLoader(dset, batch_size=args.batch_size, shuffle=True,
                                       num_workers=args.worker, drop_last=True)

    print('Confident Data Count:', len(dset), ', Confusion Matrix:')
    df_train = pd.DataFrame(dset.imgs, columns=['class', 'pseudo_label', 'true_label'])
    from sklearn.metrics import confusion_matrix
    print(confusion_matrix(df_train['true_label'], df_train['pseudo_label']))

    return confident_dset_loader


def train_source(args):
    SEED = args.seed
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
    np.random.seed(SEED)
    random.seed(SEED)

    dset_loaders = data_load(args)
    ## set base network
    model = utils.get_model(args.net, args.num_classes)
    print(args.is_pretrained_loading)
    if args.is_pretrained_loading:
        print("loading pretrained model ......")
        is_model_state_dict_valid = model.load_state_dict(torch.load(args.pretrained_model_path))
        print(args.pretrained_model_path)
        print(is_model_state_dict_valid)
    net = nn.DataParallel(model)

    # print(is_model_state_dict_valid)
    optimizer = optim.SGD(net.parameters(), lr=args.lr)
    optimizer = op_copy(optimizer)

    acc_init = 0
    auc_init = 0
    iter_per_epoch = len(dset_loaders["train_x"])
    max_iter = args.max_epoch * len(dset_loaders["train_x"])
    interval_iter = max_iter // 10
    interval_epoch = 6
    iter_num = 0

    net.train()

    losses = []
    losses_train = []
    focal_loss_class_proportion = FocalLossClassProportion(args.num_classes, args.class_proportion).cuda()
    uncertainty_loss = UncertaintyLoss(args.device, 0.8, 0.3)
    unlabeled_loss_fn = F.cross_entropy
    labeled_loss_fn = F.cross_entropy
    confident_loader = None
    while iter_num < max_iter:
        epoch = int(iter_num / iter_per_epoch)

        try:
            inputs_x, labels_x = iter_x.next()
        except:
            iter_x = iter(dset_loaders["train_x"])
            inputs_x, labels_x = iter_x.next()
        inputs_x, labels_x = inputs_x.cuda(), labels_x.cuda()

        if iter_num == iter_per_epoch * args.start_u or epoch >= args.start_u:
            # if iter_num % interval_iter == 0:
            if iter_num % (interval_epoch * iter_per_epoch) == 0:
                net.eval()
                confident_loader = obtain_confident_loader(dset_loaders["train_u"], net, args)
                net.train()

            try:
                inputs_c, labels_c, real_c, _ = iter_c.next()
            except:
                if confident_loader is None:
                    confident_loader = obtain_confident_loader(dset_loaders["train_u"], net, args)
                iter_c = iter(confident_loader)
                inputs_c, labels_c, real_c, _ = iter_c.next()

            inputs_c, labels_c, real_c = inputs_c.cuda(), labels_c.cuda(), real_c.cuda()

            # if inputs_c.size(0) % 2 == 0:
            if not args.is_test_baselines:
                logits_c, afm_logits_c = net(inputs_c, afm=True)
                loss_c = unlabeled_loss_fn(afm_logits_c, labels_c)
            else:
                features_c, logits_c = net(inputs_c)
                loss_c = F.cross_entropy(logits_c, labels_c)

            _, preds_c = torch.max(logits_c.data, 1)
            num_correct_c = torch.sum(preds_c == real_c.data)
            running_corrects_c = num_correct_c.float() / float(preds_c.shape[0]) * 100

        # AFM
        if not args.is_test_baselines:
            logits_train, afm_logits_train = net(inputs_x, afm=True)
            loss_train = args.weight_naive * labeled_loss_fn(
                afm_logits_train, labels_x) + args.weight_afm * labeled_loss_fn(logits_train, labels_x)
            losses_train.append(loss_train.item())
        else:
            features_train, logits_train = net(inputs_x)
            loss_train = F.cross_entropy(logits_train, labels_x)
            losses_train.append(loss_train.item())

        # Running Accuracy
        _, preds_x = torch.max(logits_train.data, 1)
        num_correct_x = torch.sum(preds_x == labels_x.data)
        running_corrects_x = num_correct_x.float() / float(preds_x.shape[0]) * 100

        args.num_eval = iter_num
        if epoch < args.start_u:
            loss = loss_train
            print('epoch:{}/{}, iter:{}/{}, loss_train: {:.2f}, acc_x: {:.2f}%'
                  .format(epoch + 1, args.max_epoch, iter_num, max_iter, loss_train.item(), running_corrects_x.item()))
            args.writer.add_scalar("train/1.loss_train", loss_train.item(), args.num_eval)
            args.writer.add_scalar("train/2.acc_x", running_corrects_x.item(), args.num_eval)
        else:
            loss = loss_train + args.weight_u * loss_c
            print('epoch:{}/{}, iter:{}/{}, loss_train: {:.2f}, loss_c: {:.2f}, acc_x: {:.2f}%, acc_u: {:.2f}'
                  .format(epoch + 1, args.max_epoch, iter_num, max_iter, loss_train.item(), loss_c.item(), \
                          running_corrects_x.item(), running_corrects_c.item()))
            args.writer.add_scalar("train/1.loss_train", loss_train.item(), args.num_eval)
            args.writer.add_scalar("train/2.loss_c", loss_c.item(), args.num_eval)
            args.writer.add_scalar("train/3.acc_x", running_corrects_x.item(), args.num_eval)
            args.writer.add_scalar("train/4.acc_u", running_corrects_c.item(), args.num_eval)

        losses.append(loss.item())
        iter_num += 1

        utils.lr_scheduler(optimizer, iter_num=iter_num, max_iter=max_iter)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # if iter_num % interval_iter == 0 or iter_num == max_iter:
        if iter_num % (interval_epoch * iter_per_epoch) == 0 or iter_num == max_iter:
            net.eval()
            features, logits, y_true, y_predict = get_test_data(dset_loaders['test'], net)

            accuracy, kappa, report, sensitivity, specificity, roc_auc, f1, recall, precision = \
                                                            get_metrics_sev_class(logits, y_true, y_predict)

            if args.num_classes == 2:
                accuracy, kappa, report, sensitivity, specificity, roc_auc = get_metrics(logits, y_true, y_predict)
                log_str = 'Epoch:{}/{}, Iter:{}/{}; Accuracy = {:.2f}%, Kappa = {:.4f},' \
                          ' Sensitivity = {:.4f}, Specificity = {:.4f}, AUROC = {:.4f}' \
                          'F1 = {:.4f}, Precision = {:.4f}, Recall = {:.4f}' \
                    .format(epoch + 1, args.max_epoch, iter_num, max_iter, accuracy,
                            kappa, sensitivity, specificity, roc_auc, f1, precision, recall)
                log_str_report = 'Report:{:.2f}%,{:.4f},{:.4f},{:.4f},{:.4f}' \
                    .format(accuracy, kappa, sensitivity, specificity, roc_auc)
                args.writer.add_scalar("test/1.Accuracy", accuracy, args.num_eval)
                args.writer.add_scalar("test/2.Kappa", kappa, args.num_eval)
            else:
                log_str = 'Epoch:{}/{}, Iter:{}/{}; Accuracy = {:.2f}%, Kappa = {:.4f}, ' \
                          'F1 = {:.4f}, Recall = {:.4f}, Precision = {:.4f}'.format(
                    epoch + 1, args.max_epoch, iter_num, max_iter, accuracy, kappa, f1, precision, recall)
                args.writer.add_scalar("test/1.Accuracy", accuracy, args.num_eval)
                args.writer.add_scalar("test/2.Kappa", kappa, args.num_eval)
                args.writer.add_scalar("test/2.F1", f1, args.num_eval)
                args.writer.add_scalar("test/2.Precision", precision, args.num_eval)
                args.writer.add_scalar("test/2.Recall", recall, args.num_eval)
                log_str_report = 'Report:{:.2f}%,{:.4f},{:.4f},{:.4f},{:.4f}' \
                    .format(accuracy, kappa, f1, precision, recall)

            args.out_file.write(log_str + '\n')
            args.out_file.write(log_str_report + '\n')
            args.out_file.flush()
            print(log_str + '\n')
            if args.is_save:
                if args.num_classes != 100:
                    if roc_auc >= auc_init:
                        torch.save(net.module.state_dict(), osp.join(args.output_dir_train, "best_params_auc.pt"))
                if args.num_classes != 100:
                    if accuracy >= acc_init:
                        acc_init = accuracy
                        torch.save(net.module.state_dict(), osp.join(args.output_dir_train, "best_params.pt"))

            net.train()

    with open(osp.join(args.output_dir_train, 'losses_train.txt'), "w") as fp:
        json.dump(losses_train, fp)

    return net


def print_args(args):
    s = "==========================================\n"
    for arg, content in args.__dict__.items():
        s += "{}:{}\n".format(arg, content)
    return s


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='My Classification')
    parser.add_argument('--name', type=str, default="afm-resnet", help='experiment name')
    parser.add_argument('--src-dset-path', type=str,
                        default='data/semi_processed',
                        help='source dataset path')
    parser.add_argument('--check_points_path', type=str, default="ckps_default",
                        help='path to save checkpoints.')
    parser.add_argument('--gpu_ids', type=str, nargs='?', default='0,1,2,3,4,5,6,7', help="device id to run")
    # parser.add_argument('--optimizer', type=str, default='sgd', choices=['sgd', 'adam'], help="device id to run")
    parser.add_argument('--labeled_num', type=int, default=500,
                        help="number of training labeled samples[500,1000,1500,2000,2500]")
    parser.add_argument('--num_classes', type=int, default=7, help="number of classes")
    parser.add_argument('--is_save', type=bool, default=True, help="is save checkpoint")
    parser.add_argument('--pretrained_model_path', type=str,
                        default="ckps/resnet50_neg_ce_dropout_1000_1000_0.99_naive_0_afm_0.5_u_0.5/best_params.pt",
                        help="training model path")
    parser.add_argument('--is_pretrained_loading', type=bool, default=False,
                        help="whether loading pretrained model or not")
    parser.add_argument('--mix', type=bool, default=True, help="mix labeled data for afm")
    parser.add_argument('--shuffle', type=bool, default=False, help="shuffle data for afm")
    parser.add_argument('--start_u', type=int, default=6, help="epoch to start afm")
    parser.add_argument('--ce_loss', type=bool, default=True, help="extra cross entropy")

    parser.add_argument('--max_epoch', type=int, default=60, help="max iterations")
    parser.add_argument('--batch_size', type=int, default=32, help="batch_size")
    parser.add_argument('--step_size', type=int, default=15, help="batch_size")
    parser.add_argument('--interval_epoch', type=int, default=6,
                        help='interval epoch to test and add pseudo-labeled data')
    parser.add_argument('--worker', type=int, default=4, help="number of workers")
    parser.add_argument('--lr', type=float, default=1e-2, help="learning rate")
    parser.add_argument('--net', type=str, default='resnet50')
    parser.add_argument('--seed', type=int, default=2021, help="random seed")

    parser.add_argument('--is_test_baselines', type=bool, default=False,
                        help="Whether the exp is baseline test")
    parser.add_argument('--weight-naive', default=0, type=float, help='loss weight of afm labeled')
    parser.add_argument('--weight-afm', default=0.5, type=float, help='loss weight of ce labeled')
    parser.add_argument('--weight-u', default=0.5, type=float, help='loss weight of afm unlabeled')
    parser.add_argument('--threshold', type=float, default=0.99, help="threshold for confident data")

    parser.add_argument('--distance', type=str, default='cosine', choices=["euclidean", "cosine"])
    parser.add_argument('--imb', type=bool, default=False, help="imbalanced sampler")
    parser.add_argument('--suffix', type=str, default='', help="for checkpoint saving")
    args = parser.parse_args()
    print(args.max_epoch)
    print(args.weight_afm)
    print(args.weight_u)
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_ids
    args.device = torch.device('cuda', 0)
    args.suffix += '_' + str(args.labeled_num) + '_' + str(args.threshold) + '_naive_' \
                   + str(args.weight_naive) + '_afm_' + str(args.weight_afm) + '_u_' + str(args.weight_u)
    args.output_dir_train = os.path.join(args.check_points_path, args.net + "_" + args.suffix)
    print(args.output_dir_train)
    if not osp.exists(args.check_points_path):
        os.system('mkdir -p ' + args.check_points_path)
    if not osp.exists(args.output_dir_train):
        os.system('mkdir -p ' + args.output_dir_train)
    if not osp.exists(args.output_dir_train):
        os.makedirs(args.output_dir_train)

    args.output_dir_train_tb = os.path.join('./results/', args.net + "_" + args.suffix)
    args.writer = SummaryWriter(args.output_dir_train_tb)
    args.out_file = open(osp.join(args.output_dir_train, 'log.txt'), 'w')
    args.out_file.write(print_args(args) + '\n')
    args.out_file.flush()
    train_source(args)
