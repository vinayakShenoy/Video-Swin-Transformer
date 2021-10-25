import torch
from torch.utils.data import DataLoader

import mmcv
from mmcv.runner import load_checkpoint, init_dist, get_dist_info, build_optimizer, set_random_seed
from mmcv import Config, DictAction
from mmcv.parallel import collate, scatter
from mmaction.models import build_recognizer
from mmaction.datasets import build_dataset, build_dataloader
from custom_dataset_loader import RawFramesDataset

import argparse
import os.path as osp
import copy

def args_parser():
    parser = argparse.ArgumentParser(description="Arguments for train dataloader")
    parser.add_argument("--model_checkpoint", type=str, help="Location to model (.pth) file")
    parser.add_argument("--model_config", type=str, help="Locationto config file")
    parser.add_argument("--annotation_file", type=str, help="Path to annotation file")
    parser.add_argument("--data_prefix", type=str, help="Data location prefix")
    parser.add_argument("--batch_size", default=3, type=int, help="Batch size")
    parser.add_argument("--epoch", default=5, type=int, help="")
    parser.add_argument('--work-dir', help='the dir to save logs and models')
    parser.add_argument(
        '--resume-from', help='the checkpoint file to resume from')
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
    model.cls_head.fc_cls = torch.nn.Linear(in_features=768, out_features=15, bias=True)
    for param in model.parameters():
        param.requires_grad = False
    for param in model.backbone.layers[-1].parameters():
        param.requires_grad = True
    for param in model.cls_head.parameters():
        param.requires_grad = True

    return model

def prepare_dataloader(cfg, annotation_file, data_prefix, batch_size):
    datasets = [build_dataset(cfg.data.train)]
    val_dataset = copy.deepcopy(cfg.data.val)
    datasets.append(build_dataset(val_dataset))
    dataloader_setting = dict(
        videos_per_gpu=cfg.data.get('videos_per_gpu', 1) // cfg.optimizer_config.get('update_interval', 1),
        workers_per_gpu=cfg.data.get('workers_per_gpu', 1),
        num_gpus=len(cfg.gpu_ids),
        dist=False,
        seed=cfg.seed)
    dataloader_setting = dict(dataloader_setting,
                              **cfg.data.get('train_dataloader', {}))
    data_loaders = [
        build_dataloader(ds, **dataloader_setting) for ds in datasets
    ]

    return data_loaders[0]

def train_model(model,
                dataloader,
                cfg,
                distributed=False,
                validate=False,
                test=dict(test_best=False, test_last=False)):

    optimizer = build_optimizer(model, cfg.optimizer)
    for ep in range(10):
        running_loss = 0
        for (batch_idx, batch) in enumerate(dataloader):
            optimizer.zero_grad()
            batch["imgs"], batch["label"] = batch["imgs"].to("cuda"), batch["label"].to("cuda")
            loss = model(return_loss=True, **batch)
            loss["loss_cls"].backward()
            optimizer.step()
            #running_loss += loss["loss_cls"].item()
            #if batch_idx % 2
        print(loss)


def main():
    args = args_parser()

    cfg = Config.fromfile(args.model_config)

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
    if args.seed is not None:
        set_random_seed(args.seed, deterministic=args.deterministic)
    cfg.seed = args.seed
    dataloader = prepare_dataloader(cfg, args.annotation_file, args.data_prefix, args.batch_size)

    model = prepare_model(cfg, args.model_checkpoint)

    #test_option = dict(test_last=args.test_last, test_best=args.test_best)
    """train_model(model,
                dataloader,
                cfg)
    """
if __name__ == '__main__':
    main()
