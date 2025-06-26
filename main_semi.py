import argparse
import os
import torch
import random
import numpy
import yaml
import logging
import time
import sys
from os.path import join, isdir
from torch.utils.data import DataLoader
import warnings

# 忽略所有警告
warnings.filterwarnings("ignore")

# os.environ["CUDA_LAUNCH_BLOCKING"] = "1"  # gpu错误同步，有助于debug，但是运行会变慢
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  # see issue #152

parser = argparse.ArgumentParser(description='PyTorch Training')
parser.add_argument('--batch_size', default=4, type=int, help='batch size')
parser.add_argument('--LR', default=0.0001, type=float, help='initial learning rate')
parser.add_argument('--weight_decay', '--wd', default=0.0005, type=float, help='default weight decay')
parser.add_argument('--start_epoch', default=0, type=int, help='manual epoch number (useful on restarts)')
parser.add_argument('--print_freq', '-p', default=8, type=int, help='print print_freq times per epoch')
parser.add_argument('--loss_lmbda', default=None, type=float, help='hype-param of loss')
parser.add_argument('--itersize', default=1, type=int, help='iter size')
parser.add_argument("--test_epoch", default=1, type=float)
parser.add_argument("-plr", "--pretrainlr", type=float, default=0.1)

parser.add_argument('--gpu', default=None, type=str, help='GPU ID')
parser.add_argument('--stepsize', default="3", type=str, help='learning rate step size')
parser.add_argument('--maxepoch', default=10, type=int, help='number of total epochs to run')
parser.add_argument("--encoder", default=None, help="caformer-m36,Dul-M36")
parser.add_argument("--decoder", default=None, help="unet,unetp,default")
parser.add_argument("--head", default=None, help="default,aspp,atten,cofusion")
parser.add_argument("--savedir", default="tmp")
parser.add_argument("--mode", type=str, default="train", choices=["train", "test"])
parser.add_argument("--resume", type=str, default=None)
parser.add_argument("--dataset", type=str, default="BSDS")
parser.add_argument("--note", default=None)
parser.add_argument("--cfg", required=True)
parser.add_argument("--loss_hype", type=float, default=1, help=" weight hyper of the ulabel loss")
parser.add_argument("--mg", action="store_true", help="Multi-granularity edge, during test")

# parser.add_argument("--rank", type=int, default=1)

args = parser.parse_args()

args.savedir = join("output", args.savedir)
# if os.path.isdir(args.savedir) and args.mode=="test": args.savedir = args.savedir + "-test"
os.makedirs(args.savedir, exist_ok=True)

args.stepsize = [int(i) for i in args.stepsize.split("-")]
print(args.stepsize)

if args.gpu is not None:
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

random_seed = 3407
random.seed(random_seed)
torch.manual_seed(random_seed)
torch.cuda.manual_seed(random_seed)
numpy.random.seed(random_seed)

from dataset.data_process import SemiData
from model.basemodel import Basemodel
from model.nms import NMS_MODEL
from utils import Logger, get_model_parm_nums, init_log, Logger
from train import semi_train
from test import test, multiscale_test


def main(cmds):
    logger = init_log('global', logging.INFO, filename=os.path.join(args.savedir, "training_{}.log".format(str(time.time()))))
    logger.propagate = 0

    with open(args.cfg, 'r', encoding='utf-8') as file:
        cfg = yaml.safe_load(file)

    if args.loss_lmbda is None:
        args.loss_lmbda = cfg["dataset"][args.dataset]["loss_lmbda"]

    if args.encoder is None:
        assert 'Encoder' in cfg["model"].keys()
        args.encoder = cfg["model"]["Encoder"]
    if args.decoder is None:
        assert 'Decoder' in cfg["model"].keys()
        args.decoder = cfg["model"]["Decoder"]
    if args.head is None:
        assert 'Head' in cfg["model"].keys()
        args.head = cfg["model"]["Head"]

    ldataset = SemiData(args.dataset, mode="ldata", args=args, cfg=cfg)
    udataset = SemiData(args.dataset, mode="udata", args=args, cfg=cfg, nsample=len(ldataset))
    testset = SemiData(args.dataset, mode="test", args=args, cfg=cfg)
    if args.loss_lmbda is None:
        args.loss_lmbda = cfg["dataset"][args.dataset]["loss_lmbda"]

    testloader = DataLoader(testset, batch_size=1, shuffle=False)

    if "MIX" in args.decoder.upper():
        assert args.head.upper() == "MIXHEAD"

    model = Basemodel(encoder_name=args.encoder,
                      decoder_name=args.decoder,
                      head_name=args.head,
                      cfg=cfg).cuda()
    logger.info(cmds)
    logger.info(args)
    logger.info("MODEL SIZE: {}".format(get_model_parm_nums(model)))

    #
    # new_key = 'new_key'
    # if 'old_key' in original_dict:
    #     original_value = original_dict.pop('old_key')  # 移除旧键及其对应的值
    #     original_dict[new_key] = original_value  # 添加新键及原有的值

    if args.resume is not None:
        ckpt = torch.load(args.resume, map_location='cpu', weights_only=True)['state_dict']
        # ckpt["encoder.conv2.0.weight"] = ckpt.pop("encoder.conv2.1.weight")
        # ckpt["encoder.conv2.0.bias"] = ckpt.pop("encoder.conv2.1.bias")
        model.load_state_dict(ckpt)
        logger.info("load pretrained model, successfully!")

    if args.mode == "test":
        assert args.resume is not None
        test(model, testloader, save_dir=join(args.savedir, os.path.basename(args.resume).split(".")[0]), mg=args.mg)
        # if "BSDS" in args.dataset.upper():
        #     test(model, testloader, save_dir=join(args.savedir,
        #                                           os.path.basename(args.resume).split(".")[0] + "-ss"))
        #     multiscale_test(model,
        #                     testloader,
        #                     save_dir=join(args.savedir, os.path.basename(args.resume).split(".")[0] + "-ms7"))
        #     multiscale_test(model,
        #                     testloader,
        #                     save_dir=join(args.savedir, os.path.basename(args.resume).split(".")[0] + "-ms3"),
        #                     scale_num=3)
        # else:
        #     test(model, testloader, save_dir=join(args.savedir,
        #                                           os.path.basename(args.resume).split(".")[0] + "-ss"))
    else:
        ldataloader = DataLoader(ldataset, batch_size=args.batch_size, num_workers=min(args.batch_size, 4), drop_last=True, shuffle=True)
        udataloader = DataLoader(udataset, batch_size=args.batch_size, num_workers=min(args.batch_size, 4), drop_last=True, shuffle=True)

        parameters = {'pretrained.weight': [], 'pretrained.bias': [], 'nopretrained.weight': [], 'nopretrained.bias': []}

        for pname, p in model.named_parameters():
            if ("encoder.stages" in pname) or ("encoder.downsample_layers" in pname):
                # p.requires_grad = False
                if "weight" in pname:
                    parameters['pretrained.weight'].append(p)
                else:
                    parameters['pretrained.bias'].append(p)
            else:
                if "weight" in pname:
                    parameters['nopretrained.weight'].append(p)
                else:
                    parameters['nopretrained.bias'].append(p)

        optimizer = torch.optim.Adam([
            {'params': parameters['pretrained.weight'], 'lr': args.LR * args.pretrainlr, 'weight_decay': args.weight_decay},
            {'params': parameters['pretrained.bias'], 'lr': args.LR * 2 * args.pretrainlr, 'weight_decay': 0.},
            {'params': parameters['nopretrained.weight'], 'lr': args.LR * 1, 'weight_decay': args.weight_decay},
            {'params': parameters['nopretrained.bias'], 'lr': args.LR * 2, 'weight_decay': 0.},
        ], lr=args.LR, weight_decay=args.weight_decay)

        # optimizer = torch.optim.Adam(model.parameters(), lr=args.LR, weight_decay=args.weight_decay)
        # optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=args.LR,
        #                              weight_decay=args.weight_decay)

        nms_model = NMS_MODEL(thrs=cfg["dataset"][args.dataset]["thrs"], nms=cfg["dataset"]["NMS"]).cuda()
        for epoch in range(args.start_epoch, args.maxepoch):
            semi_train(ldataloader, udataloader, model, nms_model, optimizer, epoch, args, cfg, logger)
            if (epoch + 1) % args.test_epoch == 0:
                test(model, testloader, save_dir=join(args.savedir, 'epoch-%d-ss-test' % epoch))


if __name__ == '__main__':
    # import datetime
    # # 获取当前日期和时间
    # current_time = datetime.datetime.now()
    # # 将日期和时间转换为字符串格式
    # time_string = current_time.strftime("%Y-%m-%d %H:%M:%S")

    cmds = "python"
    for cmd in sys.argv:
        if " " in cmd:
            cmd = "\'" + cmd + "\'"
        cmds = cmds + " " + cmd
    main(cmds)
