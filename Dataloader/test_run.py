from dataloader import DataLoader
import torch
from mmcv import Config, DictAction
from mmaction.models import build_model
from mmcv.parallel import collate, scatter
from mmaction.models import build_recognizer
from mmcv.runner import load_checkpoint

import argparse


def run_model(model, data, device=torch.device('cuda:0')):
    if next(model.parameters()).is_cuda:
        # scatter to specified GPU
        data = scatter(data, [device])[0]
    print(data['imgs'].shape)
    with torch.no_grad():
        scores = model(return_loss=True, **data)[0]
        print(scores.shape)
    return scores


def prepare_model(config_path, checkpoint_path):
    config = Config.fromfile(config_path)
    config.model.backbone.pretrained = None
    model = build_recognizer(config.model, train_cfg=config.get('train_cfg'))

    device = 'cuda:0'  # or 'cpu'
    device = torch.device(device)

    if checkpoint_path is not None:
        load_checkpoint(model, checkpoint_path, map_location=device, strict=False)

    model.cfg = config
    model.to(device)
    model.eval()
    return model

def get_data(annotation_file, data_prefix, idx=0):
    dataloader = DataLoader(ann_file=annotation_file, data_prefix=data_prefix)
    dataloader.load_annotations()
    results = dataloader.prepare_train_frames(idx)
    data = collate([results], samples_per_gpu=1)
    return data

def args_parser():
    parser = argparse.ArgumentParser(description="Arguments for train dataloader")
    parser.add_argument("--model_checkpoint", type=str, help="Location to model (.pth) file")
    parser.add_argument("--model_config", type=str, help="Locationto config file")
    parser.add_argument("--annotation_file", type=str, help="Path to annotation file")
    parser.add_argument("--data_prefix", type=str, help="Data location prefix")

    args = parser.parse_args()
    return args

def main():
    args = args_parser()
    data = get_data(args.annotation_file, args.data_prefix)
    model = prepare_model(args.model_config, args.model_checkpoint)
    scores = run_model(model, data)

if __name__ == '__main__':
    main()