from torch import nn
def get_encoder(nm,cfg=None):
    Dulbrn = cfg["model"]["Dulbrn"]
    if "CAFORMER-M36" == nm.upper():
        from model.caformer import caformer_m36_384_in21ft1k
        encoder = caformer_m36_384_in21ft1k(pretrained=cfg["model"]["ckpt"]["caformer_m36"])
    elif "DUL-M36" == nm.upper():
        from model.caformer import caformer_m36_384_in21ft1k
        encoder = caformer_m36_384_in21ft1k(pretrained=cfg["model"]["ckpt"]["caformer_m36"],
                                            Dulbrn=Dulbrn)
    elif "DUL-S18" == nm.upper():
        from model.caformer import caformer_s18_384_in21ft1k
        encoder = caformer_s18_384_in21ft1k(pretrained=cfg["model"]["ckpt"]["caformer_s18"],
                                            Dulbrn=Dulbrn)
    elif "VGG-16" == nm.upper():
        from model.vgg import VGG16_C
        encoder = VGG16_C(pretrain=cfg["model"]["ckpt"]["vgg16"])
    elif "LCAL" == nm.upper():
        from model.localextro import LCAL
        encoder = LCAL(Dulbrn=Dulbrn)
    else:
        raise Exception("Error encoder")
    return encoder

def get_head(nm,channels):
    from model.detect_head import CoFusion_head,CSAM_head,CDCM_head,Default_head,Fusion_head,Mixhead
    if nm == "aspp":
        head = CDCM_head(channels)
    elif nm == "atten":
        head = CSAM_head(channels)
    elif nm == "cofusion":
        head = CoFusion_head(channels)
    elif nm == "fusion":
        head = Fusion_head(channels)
    elif nm == "default":
        head = Default_head(channels)
    elif nm == "Mixhead":
        head = Mixhead(channels)
    else:
        raise Exception("Error head")
    return head


def get_decoder(nm, incs, oucs=None):
    if oucs is None:
        # oucs = (32, 32, 64, 128, 256, 512)
        oucs = (32, 32, 64, 128, 384)

    if nm.upper() == "UNETP":
        from model.unetp import UnetDecoder
        decoder = UnetDecoder(incs, oucs[-len(incs):])
    elif nm.upper() == "UNET":
        from model.unet import UnetDecoder
        decoder = UnetDecoder(incs, oucs[-len(incs):])
    elif nm.upper() == "UNET_ATT":
        from model.unet_att import UnetDecoder
        decoder = UnetDecoder(incs, oucs[-len(incs):])
    ## double branch nets
    elif nm.upper() == "ATTMIX":
        from model.unet_att import UnetMix
        decoder = UnetMix(incs, oucs[-len(incs):])
    elif nm.upper() == "UNETMIX":
        from model.unet import UnetMix_U
        decoder = UnetMix_U(incs, oucs[-len(incs):])
    elif nm.upper() == "UNETPPMIX":
        from model.unetp import UnetMix_UPP
        decoder = UnetMix_UPP(incs, oucs[-len(incs):])



    elif nm.upper() == "DEFAULT":
        from model.unet import Identity
        decoder = Identity(incs, oucs[-len(incs):])

    else:
        raise Exception("Error decoder")
    return decoder,oucs[-len(incs):]
