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
    parser.add_argument("--model_config", type=str, help="Location to config file")
    parser.add_argument(
        '--gpus',
        type=int,
        help='number of gpus to use '
             '(only applicable to non-distributed training)')
    parser.add_argument(
        '--gpu-ids',
        type=int,
        nargs='+',
        help='ids of gpus to use '
             '(only applicable to non-distributed training)')
    parser.add_argument('--seed', type=int, default=None, help='random seed')
    args = parser.parse_args()
    return args


def prepare_model(cfg, args, device=torch.device('cuda:0')):
    model = build_recognizer(cfg.model,
                             train_cfg=cfg.get('train_cfg'),
                             test_cfg=cfg.get('test_cfg'))
    load_checkpoint(model, args.model_checkpoint, map_location=device, strict=False)
    model.to(device)
    model.cfg = cfg
    model.eval()
    return model.cuda()


def prepare_dataloader(cfg):
    dataset = build_dataset(cfg.data.train)
    dataloader_setting = dict(
        videos_per_gpu= cfg.data.get('videos_per_gpu', 1) // cfg.optimizer_config.get('update_interval', 1),
        workers_per_gpu= cfg.data.get('workers_per_gpu', 1),
        num_gpus=len(cfg.gpu_ids),
        shuffle=False,
        dist=False,
        seed=cfg.seed)
    dataloader_setting = dict(dataloader_setting,
                              **cfg.data.get('train_dataloader', {}))
    dataloader = build_dataloader(dataset, **dataloader_setting)
    return dataloader

def main():
    args = parse_args()
    cfg = Config.fromfile(args.model_config)
    if args.gpu_ids is not None:
        cfg.gpu_ids = args.gpu_ids
    else:
        cfg.gpu_ids = range(1) if args.gpus is None else range(args.gpus)
    cfg.seed = args.seed

    model = prepare_model(cfg, args)
    dataloader = prepare_dataloader(cfg)

    newModel = model.backbone

    for (batchIdx, batch) in enumerate(dataloader):
        frames = batch["imgs"].cuda()
        #print(frames.shape)
        frames = frames.reshape((-1,) + frames.shape[2:])
        print(frames.shape)
        output = newModel(frames)
        print(output.shape)
        break


if __name__ == '__main__':
    main()
