from os.path import join
import os
from tqdm import tqdm
import torch
from PIL import Image
import numpy as np
import scipy.io as sio
from scipy import stats
from torch.distributions import Normal, Independent

def test(model, test_loader, save_dir, mg=False):
    # print("single scale test")
    model.eval()
    for idx, (image, filename) in enumerate(tqdm(test_loader)):
        if not mg:
            with torch.no_grad():
                result = model(image.cuda())
                if type(result) is tuple:
                    result = torch.sigmoid(result[0] + result[1])
                    # outputs_dist = Independent(Normal(loc=result[0] , scale=result[1] * 0.001), 1)
                    # result = torch.sigmoid(outputs_dist.rsample())
                result = result.squeeze().cpu().numpy()

            result = np.clip(result, stats.mode(result.reshape(-1)).mode.item(), 1)
            result = (result - result.min()) / (result.max() - result.min())
            result_png = Image.fromarray((result * 255).astype(np.uint8))

            png_save_dir = os.path.join(save_dir, "png")
            mat_save_dir = os.path.join(save_dir, "mat")

            os.makedirs(png_save_dir,exist_ok=True)
            os.makedirs(mat_save_dir,exist_ok=True)

            result_png.save(join(png_save_dir, "%s.png" % filename))
            sio.savemat(join(mat_save_dir, "%s.mat" % filename), {'result': result}, do_compression=True)
        else:
            # muge = [-5, -4.5, -4, -3.5, -3, -2.5, -2, -1.5, -1, -0.5, 0, 0.5, 1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5, 5]
            muge = [-5, -4.5, -4, -3.5, -3, -2.5, -2, -1.5, -1, -0.5, 0]
            for granu in muge:
                with torch.no_grad():
                    outputs = model(image.cuda())

                mean, std = outputs

                outputs = torch.sigmoid(mean + std * granu)
                # outputs = (outputs - outputs.min()) / (outputs.max() - outputs.min())
                result = torch.squeeze(outputs.detach()).cpu().numpy()

                result = np.clip(result, stats.mode(result.reshape(-1)).mode.item(), 1)
                result = (result - result.min()) / (result.max() - result.min())

                result_png = Image.fromarray((result * 255).astype(np.uint8))

                png_save_dir = os.path.join(save_dir, str(granu), "png")
                mat_save_dir = os.path.join(save_dir, str(granu), "mat")

                # if not os.path.isdir(png_save_dir):
                os.makedirs(png_save_dir,exist_ok=True)
                # if not os.path.isdir(mat_save_dir):
                os.makedirs(mat_save_dir,exist_ok=True)
                result_png.save(join(png_save_dir, "%s.png" % filename))
                sio.savemat(join(mat_save_dir, "%s.mat" % filename), {'result': result}, do_compression=True)


# /workspace/SED/output/0905-bsds-semi/epoch-0-ss-test/png
import cv2


def multiscale_test(model, test_loader, save_dir, scale_num=7):
    png_save_dir = os.path.join(save_dir, "png")
    mat_save_dir = os.path.join(save_dir, "mat")
    if not os.path.exists(png_save_dir):
        os.makedirs(png_save_dir)
    if not os.path.exists(mat_save_dir):
        os.makedirs(mat_save_dir)

    model.eval()
    if scale_num == 7:
        print("7 scale test")
        scale = [0.5, 0.75, 1.0, 1.25, 1.5, 1.75]
    else:
        print("3 scale test")
        scale = [0.5, 1.0, 1.5]

    for idx, (image, filename) in enumerate(tqdm(test_loader)):
        image = image[0]
        image_in = image.numpy().transpose((1, 2, 0))
        _, H, W = image.shape
        multi_fuse = np.zeros((H, W), np.float32)
        for k in range(0, len(scale)):
            im_ = cv2.resize(image_in, None, fx=scale[k], fy=scale[k], interpolation=cv2.INTER_LINEAR)
            im_ = torch.from_numpy(im_.transpose((2, 0, 1))).unsqueeze(0)
            with torch.no_grad():
                result = model(im_.cuda()).squeeze().cpu().numpy()
            fuse = cv2.resize(result, (W, H), interpolation=cv2.INTER_LINEAR)
            multi_fuse += fuse
        multi_fuse = multi_fuse / len(scale)

        result_png = Image.fromarray((multi_fuse * 255).astype(np.uint8))
        result_png.save(join(png_save_dir, "%s.png" % filename))

        sio.savemat(join(mat_save_dir, "%s.mat" % filename), {'result': multi_fuse}, do_compression=True)
