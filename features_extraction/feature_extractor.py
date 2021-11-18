import torch
import argparse

import mmcv
from mmcv.runner import load_checkpoint, init_dist, get_dist_info, build_optimizer, set_random_seed
from mmcv import Config, DictAction
from mmaction.models import build_recognizer

from dataloader.dataloaderTCN import DataloaderTCN

def parse_args():
    parser = argparse.ArgumentParser(description="Arguments for Feature extraction")
    parser.add_argument("--model_checkpoint", type=str, help="Location to model (.pth) file")
    parser.add_argument("--model_config", type=str, help="Location to config file")
    parser.add_argument("--batch_size", type=int)
    parser.add_argument("--num_workers", type=int)
    parser.add_argument("--videos_per_gpu", type=int)
    parser.add_argument("--annotation_file", type=str)
    parser.add_argument("--data_prefix", type=str)
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
    print("Here1")
    model = build_recognizer(cfg.model,
                             train_cfg=cfg.get('train_cfg'),
                             test_cfg=cfg.get('test_cfg'))
    load_checkpoint(model, args.model_checkpoint, map_location=device, strict=False)
    model.to(device)
    model.cfg = cfg
    model.eval()
    return model.cuda()


def prepare_dataloader(args):
    dataloader = DataloaderTCN(args.batch_size,
                               args.num_workers,
                               args.videos_per_gpu,
                               False,
                               args.annotation_file,
                               args.data_prefix,
                               args.seed)
    return dataloader.get_loader()

def main():
    args = parse_args()
    cfg = Config.fromfile(args.model_config)
    if args.gpu_ids is not None:
        cfg.gpu_ids = args.gpu_ids
    else:
        cfg.gpu_ids = range(1) if args.gpus is None else range(args.gpus)
    cfg.seed = args.seed

    model = prepare_model(cfg, args)
    dataloader = prepare_dataloader(args)

    newModel = model.backbone

    for (batchIdx, batch) in enumerate(dataloader):
        frames = batch["imgs"].cuda()
        frames = frames.reshape((-1,) + frames.shape[2:])
        with torch.no_grad():
            output = newModel(frames)

if __name__ == '__main__':
    main()
