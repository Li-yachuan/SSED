import random
import math
import numpy as np
from PIL import Image, ImageOps, ImageFilter
import torch
from torchvision import transforms


def crop(img, lb, rsize):
    assert img.size[0] >= rsize[0] and img.size[1] >= rsize[1]
    img = img.crop(rsize)
    if lb is not None:
        lb = lb.crop(rsize)
        return img, lb
    return img


def resize(img, lb, rsize):
    img = img.resize(rsize, Image.BILINEAR)
    if lb is not None:
        lb = lb.resize(rsize, Image.BILINEAR)
        return img, lb
    return img


def normalize(img, mask=None):
    img = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])(img)
    if mask is not None:
        mask = transforms.ToTensor()(mask)
        return img, mask
    return img


def blur(img, p=0.5):
    if random.random() < p:
        sigma = np.random.uniform(0.1, 2.0)
        img = img.filter(ImageFilter.GaussianBlur(radius=sigma))
    return img


def noise(img, p=0.5):
    q = random.random() * p
    noise = torch.normal(mean=0, std=1, size=img.size())
    img = img * (1 - q) + noise * q
    return img


def cutmix(wimg, img, p=0.5, ratios=(0.2, 0.4), box_ratios=(0.3, 0.7)):
    if random.random() > p:
        return wimg, img
    b, _, h, w = img.size()
    assert b > 1

    mask_area = random.uniform(ratios[0], ratios[1]) * h * w
    box_ratio = random.uniform(box_ratios[0], box_ratios[1])

    mask_h = min(int(math.sqrt(mask_area * box_ratio / (1 - box_ratio))), h)
    mask_w = min(mask_area // mask_h, w)

    left = int(random.random() * (w - mask_w))
    top = int(random.random() * (h - mask_h))
    right = int(left + mask_w)
    bottom = int(top + mask_h)

    cuted = img[:, :, top:bottom, left:right]
    cuted = torch.cat([cuted[1:, ...], cuted[0].unsqueeze(0)], dim=0)
    img[:, :, top:bottom, left:right] = cuted

    wcuted = wimg[:, :, top:bottom, left:right]
    wcuted = torch.cat([wcuted[1:, ...], wcuted[0].unsqueeze(0)], dim=0)
    wimg[:, :, top:bottom, left:right] = wcuted
    return wimg, img


def rotate(img):
    w, h = img.size
    rate = random.random() + 1
    ow = int(w * rate)
    oh = int(h * rate)
    img = img.resize((ow, oh), Image.BILINEAR)

    if random.random() > 0.5:
        angle = random.randint(0, 360)
        img = img.rotate(angle, resample=Image.BILINEAR)

    left = (ow - w) // 2
    top = (oh - h) // 2
    right = (ow + w) // 2
    bottom = (oh + h) // 2

    img = img.crop((left, top, right, bottom))
    return img


def flip(img):
    if random.random() > 0.5:
        # 水平翻转
        img = img.transpose(Image.FLIP_LEFT_RIGHT)

    return img
