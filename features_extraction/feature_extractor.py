import torch
import argparse

import mmcv
from mmcv.runner import load_checkpoint, init_dist, get_dist_info, build_optimizer, set_random_seed
from mmcv import Config, DictAction
from mmaction.models import build_recognizer
from mmaction.datasets import build_dataset, build_dataloader


def parse_args():
    parser = argparse.ArgumentParser(description="Arguments for Feature extraction")
    parser.add_argument("--model_checkpoint", type=str, help="Location to model (.pth) file")
    parser.add_argument("--model_config", type=str, help="Locationto config file")
    args = parser.parse_args()
    return args


def prepare_model(cfg, args, device=torch.device('cuda:0')):
    model = build_recognizer(cfg.model,
                             train_cfg=cfg.get('train_cfg'),
                             test_cfg=cfg.get('test_cfg'))
    load_checkpoint(model, args.model_checkpoint, map_location=device, strict=False)
    model.cfg = cfg
    model.eval()
    return model.cuda()


def prepare_dataloader(cfg):
    dataset = build_dataset(cfg.data.test)
    dataloader_setting = dict(
        videos_per_gpu=cfg.data.get('videos_per_gpu', 1) // cfg.optimizer_config.get('update_interval', 1),
        workers_per_gpu=cfg.data.get('workers_per_gpu', 1),
        num_gpus=len(cfg.gpu_ids),
        shuffle=False,
        dist=True,
        seed=cfg.seed)
    dataloader_setting = dict(dataloader_setting,
                              **cfg.data.get('test_dataloader', {}))
    dataloader = build_dataloader(dataset, **dataloader_setting)
    return dataloader

def main():
    args = parse_args()


if __name__ == '__main__':
    main()
