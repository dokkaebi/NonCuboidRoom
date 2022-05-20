from cvnets.layers.upsample import UpSample
from cvnets.models.classification.mobilevit import MobileViT
from cvnets.models.segmentation.heads.deeplabv3 import DeeplabV3
from torch import Tensor, nn

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


class MobileViTDeeplabDetector(nn.Module):
    """ Detector variant with MobileViT Deeplab as backbone

    MobileViT expects 256x256 image inputs, outputs 160 or 640 channels.
    Heads expect 256-channel input (as trained by authors, but adjustable).
    """
    def __init__(self, opts, encoder_weights=None):
        super().__init__()
        self.encoder = MobileViT(opts)
        self.init_weights(encoder_weights=encoder_weights)
        self.decoder = DeeplabV3(opts, enc_conf=self.encoder.model_conf_dict)
        self.heads = Heads()

    def forward(self, x):
        x = self.encoder.extract_end_points_all(x)
        x = self.decoder(enc_out=x)
        x = self.heads(x)
        return x

    def init_weights(self, encoder_weights=None):
        if encoder_weights is not None:
            self.encoder.load_state_dict(encoder_weights, strict=False)



class MobileViTBackbone(nn.Module):
    """ MobileViT encoder designed to replace HRNet in Detector.
    
    Following examples in cvnets/models/detection/ssd.py and
    cvnets/models/segmentation/enc_dec.py, construct backbone
    encoder and delete unused trailing layers.

    Segmentation Snippets
        # Here's how build_segmentation_model builds its encoder
        output_stride = getattr(opts, "model.segmentation.output_stride", None)
        encoder = build_classification_model( # <-- basically just MobileViT(opts, **kwargs)
            opts=opts,
            output_stride=output_stride
        )

        # SegEncoderDecoder __init__:
        # delete layers that are not required in segmentation network
        self.encoder.classifier = None
        use_l5_exp = getattr(opts, "model.segmentation.use_level5_exp", False)
        if not use_l5_exp:
            self.encoder.conv_1x1_exp = None

    HRNet in default Detector
    Outputs four feature maps at different resolutions with C, 2C, 4C, 8C channels
    HRMerge combines those in_channels=(32, 64, 128, 256) -> out_channels=256
    So Heads expect 256-channel input


    Local inference required changing deeplabv3_mobilevit_small.yaml:
        ddp.enable: false
        model.normalization.name: batch_norm  # instead of sync_batch_norm
    Also updated NonCuboidRoom/cfg.yaml
        num_workers: 8  # instead of 16

    Something about local python version (?) - had to update a getattr
     in cvnets/models/segmentation/heads/base_seg_head.py:
    self.n_classes = getattr(opts, 'model.segmentation.n_classes', None) or 20
     and also in options/utils.py:
    collections.abc.MutableMapping instead of collections.MutableMapping

    Using DeepLabv3 segmenter from cvnets, channel count mismatch
        setup:
            self.encoder = MobileViT(opts)
            self.segmenter = SegEncoderDecoder(opts, encoder=self.encoder)
        error: in first line of Heads.forward
            Given groups=1, weight of size [256, 256, 3, 3], expected input[1, 20, 192, 320] to have 256 channels, but got 20 channels instead

    
    Successfully got some output with this config
            self.encoder = MobileViT(opts)
            self.adapter = ConvLayer(
                opts,
                in_channels=self.encoder.model_conf_dict['layer5']['out'],
                out_channels=256,
                kernel_size=1,
            )
            ...
            x = self.encoder.extract_end_points_all(x)['out_l5']
            x = self.adapter(x)
        Trouble here is we need to train the adapter layer


    Also got output with this config
            self.backbone = MobileViTBackbone(opts, weights=vit_weights)
            self.heads = Heads(in_planes=160)
            ...
            x = self.backbone(x)
            x = self.heads(x)
        Here, loading weights for heads is a problem (they were trained with 256d input)


    When dropping the final layer and using 160d outputs, sometimes get
    an assertion in models/reconstruction.py:499 (about half the time)
        assert len(pfloor) + len(pceiling) == 1

    When using the final layer with 640d outputs, rarely get singular
    matrix exception in models/reconstruction.py:340
        res = np.linalg.solve(A, B)

    When using a trainable layer (ASPP or ConvLayer) to bridge between
    160d backbone output and 256d heads inputs, rare error at
    utils.py:584 - _segs_noopt or _segs_opt has size 0
        cost1 = ((_segs_gt[:, np.newaxis] + _segs_noopt)==2).sum((2, 3))


    """

    def __init__(self, opts, weights=None) -> None:
        super().__init__()
        # construct MobileViT model
        self.encoder = MobileViT(opts)
        self.init_weights(weights=weights)

        # remove unused layers
        self.encoder.classifier = None
        self.encoder.conv_1x1_exp = None

        self.upsample = UpSample(scale_factor=8, mode="bilinear", align_corners=False)

        # I think this will be 8x8x160 at this point
        # be sure to ouptut the correct shape - Heads need 256 channels
        self.merge = HRMerge(
            in_channels=(self.encoder.model_conf_dict['layer5']['out'],))


    def forward(self, x: Tensor) -> Tensor:
        # use self.encoder.extract_end_points_all(Tensor) -> Dict
        # want to return a Tensor for use in later layers

        x = self.encoder.extract_end_points_all(x)
        x = x['out_l5']  # Bx160x12x20
        x = self.upsample(x)  # Bx160x96x160
        x = self.merge([x])  # Bx256x96x160
        return x

    def init_weights(self, weights=None):
        if weights is not None:
            self.encoder.load_state_dict(weights)


class MobileViTDetector(nn.Module):
    """ Detector variant with MobileViT backbone

    MobileViT expects 256x256 image inputs, outputs 160 or 640 channels.
    Heads expect 256-channel input (as trained by authors, but adjustable).
    """
    def __init__(self, opts, vit_weights=None, heads_weights=None):
        super().__init__()
        self.backbone = MobileViTBackbone(opts, weights=vit_weights)
        self.heads = Heads()
        self.init_weights(heads_weights=heads_weights)

    def forward(self, x):
        x = self.backbone(x)
        
        # Tried a few hacks to bridge the gap in channels without
        # additional training, but not much luck
        # x = nn.functional.pad(x, (0, 0, 0, 0, 48, 48), mode='replicate')
        # Sort of works
        # xtra = x[:,20:116,:,:]
        # x = torch.concat((xtra, x), dim=1)
        # singular matrix
        # start = x[:,:48,:,:]
        # end = x[:,112:,:,:]
        # x = torch.concat((start, x, end), dim=1)

        x = self.heads(x)
        return x

    def init_weights(self, heads_weights=None):
        # strict=False because we're loading weights from the original
        # NonCuboidRoom model with a different backbone
        if heads_weights is not None:
            self.load_state_dict(heads_weights, strict=False)
