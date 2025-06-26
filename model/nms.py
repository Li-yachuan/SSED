import torch
import torch.nn.functional as F
from torch import nn


class NMS_MODEL(nn.Module):
    def __init__(self, r=2, s=5, m=1.01, thrs=0.5, nms=True):
        super(NMS_MODEL, self).__init__()
        self.nms = nms
        self.r = int(r)
        self.s = s
        self.m = m
        self.thrs = thrs

        filter_size = 3

        generated_filters = torch.tensor([[0.25, 0.5, 0.25]])

        self.gaussian_horizontal1 = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=(1, filter_size),
                                              padding=(0, filter_size // 2), bias=False)
        self.gaussian_horizontal1.weight.data.copy_(generated_filters)
        # self.gaussian_horizontal1.bias.data.copy_(torch.tensor([0.0]))

        self.gaussian_vertical1 = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=(filter_size, 1),
                                            padding=(filter_size // 2, 0), bias=False)
        self.gaussian_vertical1.weight.data.copy_(generated_filters.T)
        # self.gaussian_vertical1.bias.data.copy_(torch.tensor([0.0]))

        filter_size = 9

        generated_filters = torch.tensor([[0.04, 0.08, 0.12, 0.16, 0.2, 0.16, 0.12, 0.08, 0.04]])
        self.gaussian_filter_horizontal = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=(1, filter_size),
                                                    padding=(0, filter_size // 2), bias=False)
        self.gaussian_filter_horizontal.weight.data.copy_(generated_filters)
        # self.gaussian_filter_horizontal.bias.data.copy_(torch.tensor([0.0]))

        self.gaussian_filter_vertical = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=(filter_size, 1),
                                                  padding=(filter_size // 2, 0), bias=False)
        self.gaussian_filter_vertical.weight.data.copy_(generated_filters.T)
        # self.gaussian_filter_vertical.bias.data.copy_(torch.tensor([0.0]))

        sobel_filter = torch.tensor([[1, 0, -1],
                                     [2, 0, -2],
                                     [1, 0, -1]])

        self.sobel_filter_horizontal = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=sobel_filter.shape,
                                                 padding=sobel_filter.shape[0] // 2, bias=False)
        self.sobel_filter_horizontal.weight.data.copy_(sobel_filter)
        # self.sobel_filter_horizontal.bias.data.copy_(torch.tensor([0.0]))

        self.sobel_filter_vertical = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=sobel_filter.shape,
                                               padding=sobel_filter.shape[0] // 2, bias=False)
        self.sobel_filter_vertical.weight.data.copy_(sobel_filter.T)
        # self.sobel_filter_vertical.bias.data.copy_(torch.tensor([0.0]))

    def forward(self, img):
        if self.nms:
            E = self.gaussian_vertical1(self.gaussian_horizontal1(img))

            E1 = self.gaussian_filter_vertical(self.gaussian_filter_horizontal(E))

            grad_x = self.sobel_filter_horizontal(E1)
            grad_y = self.sobel_filter_vertical(E1)

            grad_xy = self.sobel_filter_vertical(grad_x)
            grad_xx = self.sobel_filter_horizontal(grad_x)
            grad_yy = self.sobel_filter_vertical(grad_y)

            # grad_mag = torch.sqrt(grad_x ** 2 + grad_y ** 2)

            O = torch.atan(grad_yy * torch.sign(-grad_xy) / (grad_xx + 1e-5)) % 3.14

            ## edgesNmsMex(E, O, 2, 5, 1.01)
            E *= self.m
            coso = torch.cos(O)
            sino = torch.sin(O)
            _, _, H, W = img.size()
            norm = torch.tensor([[[[W, H]]]]).type_as(E)
            h = torch.linspace(-1.0, 1.0, H).view(-1, 1).repeat(1, W)
            w = torch.linspace(-1.0, 1.0, W).repeat(H, 1)
            grid = torch.cat((w.unsqueeze(2), h.unsqueeze(2)), 2).type_as(E)
            E0 = E.clone()
            for d in range(-self.r, self.r + 1):
                if d == 0: continue
                grid1 = grid - torch.stack((coso * d, sino * d), dim=-1).squeeze() / norm
                neighber = F.grid_sample(E, grid1, align_corners=True)

                E0[neighber > E0] = 0

            # suppress noisy edge estimates near boundaries

            s1 = 1.0 / self.s
            E0[:, :, :, :self.s] *= s1
            E0[:, :, :, W - self.s - 1:] *= s1
            E0[:, :, :self.s, :] *= s1
            E0[:, :, H - self.s - 1:, :] *= s1

            E0_2 = torch.nn.functional.conv2d(E0, torch.tensor([[[[1., 1, 1, 1, 1],
                                                                  [1, 0, 0, 0, 1],
                                                                  [1, 0, 0, 0, 1],
                                                                  [1, 0, 0, 0, 1],
                                                                  [1, 1, 1, 1, 1]]]]).cuda(), padding=2)
            E0_1 = torch.nn.functional.conv2d(E0, torch.tensor([[[[1., 1, 1],
                                                                  [1, 0, 1],
                                                                  [1, 1, 1]]]]).cuda(), padding=1)
            E0 *= torch.sign(E0_2)
            E0 *= torch.sign(E0_1)
            E0 = (E0 - E0.min()) / (E0.max() - E0.min())

            E0[E0 > self.thrs] = 1
            E0[E0 < (1 - self.thrs)] = 0
            E0[(E0 >= (1 - self.thrs)) & (E0 <= self.thrs)] = 2

            return E0
        else:
            img[img > self.thrs] = 1
            img[img < (1 - self.thrs)] = 0
            img[(img >= (1 - self.thrs)) & (img <= self.thrs)] = 2

            return img
