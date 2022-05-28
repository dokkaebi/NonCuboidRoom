from cvnets.models.classification.mobilenetv2 import MobileNetV2
from cvnets.models.classification.mobilevit import MobileViT
from cvnets.models.segmentation.heads.deeplabv3 import DeeplabV3
from torch import nn

from models.heads import Heads, HRMerge
from models.hr_cfg import model_cfg
from models.hrnet import HighResolutionNet


class Detector(nn.Module):
    def __init__(self):
        super(Detector, self).__init__()
        extra = model_cfg['backbone']['extra']
        self.backbone = HighResolutionNet(extra)
        self.merge = HRMerge()
        self.heads = Heads()
        self.init_weights(pretrained=None)

    def forward(self, x):
        x = self.backbone(x)
        x = self.merge(x)
        x = self.heads(x)
        return x

    def init_weights(self, pretrained=None):
        self.backbone.init_weights(pretrained)
        self.merge.init_weights()


class BaseDeeplabDetector(nn.Module):
    """ Detector variant with Deeplab feature extractor

    Expects 256x256 image inputs, outputs 160 or 640 channels.
    Heads expect 256-channel input (as trained by authors, but adjustable).
    """
    def __init__(self, opts, encoder):
        super().__init__()
        self.encoder = encoder
        self.decoder = DeeplabV3(opts, enc_conf=self.encoder.model_conf_dict)
        self.heads = Heads()

    def forward(self, x):
        x = self.encoder.extract_end_points_all(x)
        x = self.decoder(enc_out=x)
        x = self.heads(x)
        return x


class MobileNetDeeplabDetector(BaseDeeplabDetector):
    """ Deeplab Detector with MobileNetV2 encoder """
    def __init__(self, opts, encoder_weights=None):
        encoder = MobileNetV2(opts)
        if encoder_weights is not None:
            encoder.load_state_dict(encoder_weights)
        super().__init__(opts, encoder)


class MobileViTDeeplabDetector(BaseDeeplabDetector):
    """ Deeplab Detector with MobileViT encoder """
    def __init__(self, opts, encoder_weights=None):
        encoder = MobileViT(opts)
        if encoder_weights is not None:
            encoder.load_state_dict(encoder_weights)
        super().__init__(opts, encoder)

