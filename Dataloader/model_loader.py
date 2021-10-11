import torch
from mmcv import Config, DictAction
from mmaction.models import build_model

PATH = "models/swin_base_patch244_window877_kinetics400_22k.pth"
CFG_PATH = "configs/recognition/swin/swin_base_patch244_window877_kinetics400_22k.py"
cfg = Config.fromfile(CFG_PATH)
model = build_model(
        cfg.model, train_cfg=None, test_cfg=cfg.get('test_cfg'))
new_model = model.backbone




