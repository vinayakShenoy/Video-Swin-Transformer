from torch.utils.data import DataLoader
from .rawframe_dataset import RawFramesDataset
from functools import partial
from mmcv.parallel import collate
from mmcv.runner import get_dist_info
import numpy as np, random

class DataloaderTCN():
    def __init__(self,
                 batch_size,
                 num_workers,
                 videos_per_gpu,
                 shuffle,
                 annotation_file,
                 data_prefix,
                 seed=None):
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.videos_per_gpu = videos_per_gpu
        self.shuffle = shuffle
        self.annotation_file = annotation_file
        self.data_prefix = data_prefix
        self.seed = seed
        self.rank, self.world_size = get_dist_info()

    def get_loader(self):
        dataset = RawFramesDataset(annotations_file=self.annotation_file,
                                   img_dir=self.data_prefix)
        init_fn = partial(
            self.worker_init_fn, num_workers=self.num_workers, rank=self.rank,
            seed=self.seed) if self.seed is not None else None
        data_loader = DataLoader(
            dataset,
            batch_size=1,
            num_workers=0,#self.num_workers,
            collate_fn=None,#partial(collate, samples_per_gpu=self.videos_per_gpu),
            pin_memory=False,
            shuffle=False,
            worker_init_fn=None,#init_fn,
            drop_last=False)
        return data_loader

    def worker_init_fn(self, worker_id, num_workers, rank, seed):
        """Init the random seed for various workers."""
        # The seed of each worker equals to
        # num_worker * rank + worker_id + user_seed
        worker_seed = num_workers * rank + worker_id + seed
        np.random.seed(worker_seed)
        random.seed(worker_seed)