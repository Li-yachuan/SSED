from torch.utils import data
from os.path import join, abspath, splitext, split, isdir, isfile, basename, dirname
from PIL import Image
from torchvision import transforms
import torch
import numpy as np
import random
import imageio
from pathlib import Path
from torch.nn.functional import interpolate
from dataset.transformer import *
import math


class BSDS_VOCLoader(data.Dataset):
    """
    Dataloader BSDS500
    """

    def __init__(self, root='data/HED-BSDS_PASCAL', split='train',
                 transform=False, threshold=0.3, ablation=False):
        self.root = root
        self.split = split
        self.threshold = threshold * 256
        print('Threshold for ground truth: %f on BSDS_VOC' % self.threshold)
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            normalize])
        if self.split == 'train':

            self.filelist = join(root, "HED-BSDS/bsds_pascal_train_pair.lst")
        elif self.split == 'test':

            self.filelist = join(root, "HED-BSDS/test.lst")
        else:
            raise ValueError("Invalid split type!")
        with open(self.filelist, 'r') as f:
            self.filelist = f.readlines()

    def __len__(self):
        return len(self.filelist)

    def __getitem__(self, index):
        if self.split == "train":
            img_file, lb_file = self.filelist[index].split()
            img_file = img_file.strip()
            lb_file = lb_file.strip()
            lb = np.array(Image.open(join(self.root, lb_file)), dtype=np.float32)
            if lb.ndim == 3:
                lb = np.squeeze(lb[:, :, 0])
            assert lb.ndim == 2
            threshold = self.threshold
            lb = lb[np.newaxis, :, :]
            lb[lb == 0] = 0
            lb[np.logical_and(lb > 0, lb < threshold)] = 2
            lb[lb >= threshold] = 1

        else:
            img_file = self.filelist[index].rstrip()

        with open(join(self.root, img_file), 'rb') as f:
            img = Image.open(f)
            img = img.convert('RGB')
        img = self.transform(img)

        if self.split == "train":
            return img, lb
        else:
            img_name = Path(img_file).stem
            return img, img_name


class BSDS_Loader(data.Dataset):
    """
    Dataloader BSDS500
    """

    def __init__(self, root='data/HED-BSDS', split='train', threshold=0.3, colorJitter=False):
        self.root = root
        self.split = split
        self.threshold = threshold
        print('Threshold for ground truth: %f on BSDS' % self.threshold)
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])
        if colorJitter:
            self.transform = transforms.Compose([
                transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.5),
                transforms.RandomGrayscale(p=0.2),
                transforms.ToTensor(),
                normalize])
        else:
            self.transform = transforms.Compose([
                transforms.ToTensor(),
                normalize])
        if self.split == 'train':
            self.filelist = join(self.root, 'train_BSDS.lst')
        elif self.split == 'test':
            self.filelist = join(self.root, 'test.lst')
        else:
            raise ValueError("Invalid split type!")
        with open(self.filelist, 'r') as f:
            self.filelist = f.readlines()

    def __len__(self):
        return len(self.filelist)

    def __getitem__(self, index):
        if self.split == "train":
            img_lb_file = self.filelist[index].strip("\n").split(" ")
            img_file = img_lb_file[0]

            lb_file = img_lb_file[random.randint(1, len(img_lb_file) - 1)]
            lb = transforms.ToTensor()(Image.open(join(self.root, lb_file)))

        else:
            img_file = self.filelist[index].rstrip()

        with open(join(self.root, img_file), 'rb') as f:
            img = Image.open(f)
            img = img.convert('RGB')
        img = self.transform(img)

        if self.split == "train":
            return img, lb
        else:
            img_name = Path(img_file).stem
            return img, img_name


class BIPED_Loader(data.Dataset):
    """
    Dataloader BSDS500
    """

    def __init__(self, root=' ', split='train', transform=False):
        self.root = root
        self.split = split
        self.transform = transform
        if self.split == 'train':
            self.filelist = join(root, "train_pair.lst")

        elif self.split == 'test':
            self.filelist = join(root, "test.lst")
        else:
            raise ValueError("Invalid split type!")
        with open(self.filelist, 'r') as f:
            self.filelist = f.readlines()
        # print(self.filelist)

    def __len__(self):
        return len(self.filelist)

    def __getitem__(self, index):
        if self.split == "train":
            img_file, lb_file = self.filelist[index].strip("\n").split(" ")

        else:
            img_file = self.filelist[index].rstrip()

        img = imageio.imread(join(self.root, img_file))
        img = transforms.ToTensor()(img)

        if self.split == "train":
            label = transforms.ToTensor()(imageio.imread(join(self.root, lb_file), as_gray=True)) / 255
            img, label = self.crop(img, label)
            label[label >= 0.5] = 1
            label[label < 0.5] = 0
            assert torch.all((label == 0) | (label == 1)), "label包含除0和1之外的值"
            return img, label

        else:
            img_name = Path(img_file).stem
            return img, img_name

    @staticmethod
    def crop(img, lb):
        _, h, w = img.size()
        assert (h > 400) and (w > 400)
        crop_size = 400
        i = random.randint(0, h - crop_size)
        j = random.randint(0, w - crop_size)
        img = img[:, i:i + crop_size, j:j + crop_size]
        lb = lb[:, i:i + crop_size, j:j + crop_size]

        return img, lb


class PASCAL_Loader(data.Dataset):
    """
    Dataloader BSDS500
    """

    def __init__(self, root='data/HED-BSDS'):
        self.root = root
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            normalize])

        self.filelist = join(self.root, 'train_PASCAL.lst')

        with open(self.filelist, 'r') as f:
            self.filelist = f.readlines()

    def __len__(self):
        return len(self.filelist)

    def __getitem__(self, index):
        img_file, lb_file = self.filelist[index].strip("\n").split(" ")
        lb = Image.open(join(self.root, lb_file)).convert("L")
        lb = transforms.ToTensor()(lb)
        img = Image.open(join(self.root, img_file)).convert('RGB')
        img = self.transform(img)
        return img, lb


class SemiData(data.Dataset):

    def __init__(self, name, mode, cfg, args, nsample=None):
        self.dataset = name
        self.mode = mode
        self.root = cfg["dataset"][args.dataset]["root"]
        with open(join(self.root, cfg["dataset"][name][mode]), 'r') as f:
            self.filelist = f.readlines()


        if self.mode == "udata" and name == "SemiBSDS":
            self.filelist = self.filelist[:len(self.filelist) // 2]
            print("use the first half unflip data of SemiBSDS, and random flip them during loading")
            assert len(self.filelist) == 10103

        if self.mode == "ldata" and name == "SemiNYUD":
            # self.filelist = [pth for pth in self.filelist if "Images_05" not in pth]
            # print("do not use the 0.5 scale data", len(self.filelist))
            # assert len(self.filelist) == 12720
            assert len(self.filelist) == 19080

        if nsample:
            random.shuffle(self.filelist)
            self.filelist *= math.ceil(nsample / len(self.filelist))
            self.filelist = self.filelist[:nsample]

        self.ColorJitter = cfg["dataset"]["ColorJitter"]
        self.RandomGray = cfg["dataset"]["RandomGray"]
        self.GaussianBlur = cfg["dataset"]["GaussianBlur"]
        self.GaussianNoise = cfg["dataset"]["GaussianNoise"]
        self.Rotate = cfg["dataset"]["Rotate"]
        self.Flip = cfg["dataset"]["Flip"]

        self.label_method = cfg["dataset"]["label_method"]
        self.rsize = tuple(cfg["dataset"][args.dataset]["crop_size"])

    def __len__(self):
        return len(self.filelist)

    def crop(self, img, lb):
        _, h, w = img.size()
        assert (h >= self.rsize[1]) and (w >= self.rsize[0]), f"Imagesize:({h},{w}), cropsize({self.rsize[1]},{self.rsize[0]})"
        i = random.randint(0, h - self.rsize[1])
        j = random.randint(0, w - self.rsize[0])
        img = img[:, i:i + self.rsize[1], j:j + self.rsize[0]]
        lb = lb[:, i:i + self.rsize[1], j:j + self.rsize[0]]

        return img, lb

    def __getitem__(self, index):
        img_lb_file = self.filelist[index].strip("\n").split(" ")
        img_file = img_lb_file[0]
        img = Image.open(join(self.root, img_file)).convert('RGB')

        if self.mode == "test":
            return normalize(img), basename(img_file).split('.')[0]
        elif self.mode == "ldata":

            if len(img_lb_file) == 2:  # only one label
                lb = Image.open(join(self.root, img_lb_file[-1])).convert('L')
                img, lb = normalize(img, lb)
                lb[lb > 0.4] = 1
                lb[lb <= 0.4] = 0
                assert torch.all((lb == 0) | (lb == 1)), "label包含除0和1之外的值"
            elif self.label_method == "mix":  # multi label and mix them
                lb = [transforms.ToTensor()(Image.open(join(self.root, l))) for l in img_lb_file[1:]]
                lb = torch.mean(torch.stack(lb, dim=0), dim=0, keepdim=False)
                lb[lb >= 0.3] = 1
                lb[(lb < 0.3) & (lb > 0)] = 2
                img = normalize(img)
                assert torch.all((lb == 0) | (lb == 1) | (lb == 2)), "label包含除0和1,2之外的值"
            else:  # multi label and randm select them
                lb_file = img_lb_file[random.randint(1, len(img_lb_file) - 1)]
                lb = Image.open(join(self.root, lb_file)).convert('L')
                img, lb = normalize(img, lb)
                lb[lb > 0.4] = 1
                lb[lb <= 0.4] = 0
                assert torch.all((lb == 0) | (lb == 1)), "label包含除0和1之外的值"

            img, lb = self.crop(img, lb)
            return img, lb
        elif self.mode == "udata":
            if self.Flip:
                img = flip(img)
            if self.Rotate:
                img = rotate(img)
            if img.size != self.rsize:
                img = resize(img, None, self.rsize)
            wimg = normalize(img)
            if self.ColorJitter and random.random() < 0.8:
                img = transforms.ColorJitter(0.5, 0.5, 0.5, 0.25)(img)
            if self.RandomGray:
                img = transforms.RandomGrayscale(p=0.2)(img)
            if self.GaussianBlur:
                img = blur(img, p=0.5)
            img = normalize(img)
            if self.GaussianNoise:
                img = noise(img, p=0.6)

            wimg, img = self.crop(wimg, img)
            return wimg, img
