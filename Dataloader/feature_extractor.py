from mmaction.datasets import build_dataset, build_dataloader

from mmcv import Config, DictAction

import argparse
import copy

def args_parser():
    parser = argparse.ArgumentParser(description="Arguments for train dataloader")
    parser.add_argument("--model_checkpoint", type=str, help="Location to model (.pth) file")
    parser.add_argument("--model_config", type=str, help="Locationto config file")

def prepare_dataloader(cfg):
    datasets = [build_dataset(cfg.data.train)]

    val_dataset = copy.deepcopy(cfg.data.val)
    datasets.append(build_dataset(val_dataset))
    dataloader_setting = dict(
        videos_per_gpu=cfg.data.get('videos_per_gpu', 1) // cfg.optimizer_config.get('update_interval', 1),
        workers_per_gpu=cfg.data.get('workers_per_gpu', 1),
        num_gpus=len(cfg.gpu_ids),
        dist=True,
        seed=cfg.seed)
    dataloader_setting = dict(dataloader_setting,
                              **cfg.data.get('train_dataloader', {}))
    data_loaders = [
        build_dataloader(ds, **dataloader_setting) for ds in datasets
    ]

    return data_loaders[0]

def main():
    args = args_parser()

    cfg = Config.fromfile(args.model_config)

if __name__ == '__main__':
    main()