from dataloader import DataLoader
import torch
from mmcv import Config, DictAction
from mmaction.models import build_model
from mmcv.parallel import collate, scatter
from mmaction.models import build_recognizer
from mmcv.runner import load_checkpoint

# Get checkpoint of model parameters
CHECKPOINT_PATH = "../models/swin_base_patch244_window877_kinetics400_22k.pth"

# Get config file that contains model config and pipeline
CFG_PATH = "../configs/recognition/swin/swin_base_patch244_window877_kinetics400_22k.py"
ANN_FILE = "../data/kinetics400/kinetics400_val_list_rawframes.txt"
DATA_PREFIX = "../data/kinetics400/rawframes_val/"

dataloader = DataLoader(ann_file=ANN_FILE, data_prefix=DATA_PREFIX)
dataloader.load_annotations()
results = dataloader.prepare_train_frames(0)
data = collate([results], samples_per_gpu=1)

config = Config.fromfile(CFG_PATH)
config.model.backbone.pretrained = None
model = build_recognizer(config.model, test_cfg=config.get('test_cfg'))

device = 'cuda:0' # or 'cpu'
device = torch.device(device)
if CHECKPOINT_PATH is not None:
    load_checkpoint(model, CHECKPOINT_PATH, map_location=device, strict=False)

model.cfg = config
model.to(device)
model.eval() # for inference
#backbone_model = model.backbone

if next(model.parameters()).is_cuda:
    # scatter to specified GPU
    data = scatter(data, [device])[0]

with torch.no_grad():
    scores = model(return_loss=False, **data)[0]

print(scores.shape)