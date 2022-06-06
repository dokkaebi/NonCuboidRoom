import torch
from models.detector import *
from torch.utils.mobile_optimizer import optimize_for_mobile
from options.opts import get_training_arguments
from options.utils import load_config_file

if __name__ == '__main__':
    ex = torch.rand(1, 3, 384, 640)
    models = {}
    
    m = Detector()
    s = torch.load('checkpoints/Structured3D_pretrained.pt', map_location=torch.device('cpu'))
    m.load_state_dict(s)
    models['hrnet'] = m


    parser = get_training_arguments(parse_args=False)
    opts = parser.parse_args()
    setattr(opts, 'common.config_file', 'config/deeplabv3_mobilevit_small.yaml')
    opts = load_config_file(opts)
    dlab = MobileViTDeeplabDetector(opts)
    dlab.eval()
    state_dict = torch.load('checkpoints/checkpoints_dlab/best.pt',
                            map_location=torch.device('cpu'))
    dlab.load_state_dict(state_dict)
    models['dlab'] = dlab


    parser = get_training_arguments(parse_args=False)
    opts = parser.parse_args()
    setattr(opts, 'common.config_file', 'config/deeplabv3_mobilenetv2.yaml')
    opts = load_config_file(opts)
    mobilenet = MobileNetDeeplabDetector(opts)
    mobilenet.eval()
    state_dict = torch.load('checkpoints/checkpoints_mobilenet/best.pt',
                            map_location=torch.device('cpu'))
    mobilenet.load_state_dict(state_dict)
    models['mobilenet'] = mobilenet


    for name, model in models.items():
        # strict=False to allow model to output a dict
        tm = torch.jit.trace(model, ex, strict=False)
        opt = optimize_for_mobile(tm)
        opt._save_for_lite_interpreter(f'{name}_mobile.pt')
