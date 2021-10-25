import torch
from torch.utils.data import DataLoader

import mmcv
from mmcv.runner import load_checkpoint, init_dist, get_dist_info, build_optimizer
from mmcv import Config, DictAction
from mmcv.parallel import collate, scatter
from mmaction.models import build_recognizer

from custom_dataset_loader import RawFramesDataset

import argparse
import os.path as osp

def args_parser():
    parser = argparse.ArgumentParser(description="Arguments for train dataloader")
    parser.add_argument("--model_checkpoint", type=str, help="Location to model (.pth) file")
    parser.add_argument("--model_config", type=str, help="Locationto config file")
    parser.add_argument("--annotation_file", type=str, help="Path to annotation file")
    parser.add_argument("--data_prefix", type=str, help="Data location prefix")
    parser.add_argument("--batch_size", default=3, type=int, help="Batch size")
    parser.add_argument("--epoch", default=5, type=int, help="")
    '''
    parser.add_argument('--work-dir', help='the dir to save logs and models')
    parser.add_argument(
        '--validate',
        action='store_true',
        help='whether to evaluate the checkpoint during training')
    parser.add_argument(
        '--test-last',
        action='store_true',
        help='whether to test the checkpoint after training')
    parser.add_argument(
        '--test-best',
        action='store_true',
        help=('whether to test the best checkpoint (if applicable) after '
              'training'))
    group_gpus = parser.add_mutually_exclusive_group()
    group_gpus.add_argument(
        '--gpus',
        type=int,
        help='number of gpus to use '
             '(only applicable to non-distributed training)')
    group_gpus.add_argument(
        '--gpu-ids',
        type=int,
        nargs='+',
        help='ids of gpus to use '
             '(only applicable to non-distributed training)')
    parser.add_argument('--seed', type=int, default=None, help='random seed')
    parser.add_argument(
        '--deterministic',
        action='store_true',
        help='whether to set deterministic options for CUDNN backend.')
    parser.add_argument(
        '--cfg-options',
        nargs='+',
        action=DictAction,
        default={},
        help='override some settings in the used config, the key-value pair '
             'in xxx=yyy format will be merged into config file. For example, '
             "'--cfg-options model.backbone.depth=18 model.backbone.with_cp=True'")
    parser.add_argument(
        '--launcher',
        choices=['none', 'pytorch', 'slurm', 'mpi'],
        default='none',
        help='job launcher')
    parser.add_argument('--local_rank', type=int, default=0)
    '''
    args = parser.parse_args()
    return args


def prepare_model(cfg, checkpoint_path, device=torch.device('cuda:0')):

    model = build_recognizer(cfg.model,
                             train_cfg=cfg.get('train_cfg'),
                             test_cfg=cfg.get('test_cfg'))

    if checkpoint_path is not None:
        load_checkpoint(model, checkpoint_path, map_location=device, strict=False)

    model.cfg = cfg
    model.to(device)
    return model


def prepare_dataloader(annotation_file, data_prefix, batch_size):
    dataset = RawFramesDataset(annotations_file=annotation_file,
                               img_dir=data_prefix)
    dataloader = DataLoader(dataset,
                            batch_size=batch_size,
                            shuffle=True,
                            num_workers=1)
    return dataloader

'''
def train_model(model,
                dataloader,
                cfg,
                distributed=False,
                validate=False,
                test=dict(test_best=False, test_last=False)):

    optimizer = build_optimizer(model, cfg.optimizer)
'''
def main():
    args = args_parser()

    cfg = Config.fromfile(args.model_config)
    '''
    if cfg.get('cudnn_benchmark', False):
        torch.backends.cudnn.benchmark = True
    cfg.model.backbone.pretrained = None

    if args.work_dir is not None:
        # update configs according to CLI args if args.work_dir is not None
        cfg.work_dir = args.work_dir
    elif cfg.get('work_dir', None) is None:
        # use config filename as default work_dir if cfg.work_dir is None
        cfg.work_dir = osp.join('./work_dirs',
                                osp.splitext(osp.basename(args.config))[0])
    if args.resume_from is not None:
        cfg.resume_from = args.resume_from
    if args.gpu_ids is not None:
        cfg.gpu_ids = args.gpu_ids
    else:
        cfg.gpu_ids = range(1) if args.gpus is None else range(args.gpus)

        # init distributed env first, since logger depends on the dist info.
    if args.launcher == 'none':
        distributed = False
    else:
        distributed = True
        init_dist(args.launcher, **cfg.dist_params)
        _, world_size = get_dist_info()
        cfg.gpu_ids = range(world_size)

        # The flag is used to determine whether it is omnisource training
    cfg.setdefault('omnisource', False)

    # The flag is used to register module's hooks
    cfg.setdefault('module_hooks', [])

    # create work_dir
    mmcv.mkdir_or_exist(osp.abspath(cfg.work_dir))
    '''
    dataloader = prepare_dataloader(args.annotation_file, args.data_prefix, args.batch_size)

    model = prepare_model(args.model_config, args.model_checkpoint)

    test_option = dict(test_last=args.test_last, test_best=args.test_best)

    for (batch_idx, batch)  in enumerate(dataloader):
        batch = collate([batch], samples_per_gpu=1)
        scores = model(return_loss=True, **batch)[0]


if __name__ == '__main__':
    main()
