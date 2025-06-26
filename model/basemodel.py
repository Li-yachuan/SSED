from torch import nn
from model.utils import get_encoder, get_decoder, get_head
import random


class Basemodel(nn.Module):
    def __init__(self,
                 encoder_name="caformer-m36",
                 decoder_name="unetp",
                 head_name=None,
                 cfg=None):
        super(Basemodel, self).__init__()
        self.encoder = get_encoder(encoder_name, cfg=cfg)

        self.decoder, self.decoder_channel = get_decoder(decoder_name, self.encoder.out_channels)

        self.head = get_head(head_name, self.decoder_channel)

        self.drop = nn.Dropout2d(0.5) if cfg["model"]["FeatDrop"] else nn.Identity()

    def forward(self, x):
        _, _, H, W = x.size()
        features = self.encoder(x)
        if random.random() > 0.5:
            features = [self.drop(f) for f in features]
        features = self.decoder(features)
        edge = self.head(features)
        return edge
        # return interpolate(edge, (H, W), mode="bilinear")
