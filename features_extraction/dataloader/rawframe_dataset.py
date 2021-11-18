import pandas as pd
from torch.utils.data import Dataset
import os

from .sample_frames import SampleFrames
from .raw_frame_decode import RawFrameDecode
from .three_crop import ThreeCrop
#from random_resized_crop import RandomResizedCrop
from .resize import Resize
from .collect import Collect
from .flip import Flip
from .normalize import Normalize
from .to_tensor import ToTensor
from .format_shape import FormatShape


class RawFramesDataset(Dataset):
    def __init__(self, annotations_file, img_dir):
        self.raw_frame_ann = pd.read_csv(annotations_file, header=None)
        self.img_dir = img_dir
        self.start_index = 0
        self.modality = 'RGB'
        self.filename_tmpl = 'img_{:05}.jpg'

    def __len__(self):
        return len(self.raw_frame_ann)

    def __getitem__(self, idx):
        raw_frame_path = os.path.join(self.img_dir, self.raw_frame_ann.iloc[idx, 0])
        frames_path, num_frames, label = raw_frame_path.split()
        results = dict()
        results['frame_dir'] = frames_path
        results['total_frames'] = int(num_frames)
        results['label'] = int(label)
        results['filename_tmpl'] = self.filename_tmpl
        results['modality'] = self.modality
        results['start_index'] = self.start_index
        results = self.perform_transforms(results)
        return {"imgs": results['imgs']}

    def perform_transforms(self, data):
        img_norm_cfg = dict(
            mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_bgr=False)
        transforms = [
            SampleFrames(clip_len=32,
                         feature_extraction=True),
            RawFrameDecode(),
            Resize(scale=(-1, 224)),
            Resize(scale=(224, 224), keep_ratio=False),
            Flip(flip_ratio=0),
            Normalize(**img_norm_cfg),
            FormatShape(input_format='NCTHW'),
            Collect(keys=['imgs', 'frame_dir'], meta_keys=[]),
            ToTensor(keys=['imgs'])
        ]

        for t in transforms:
            data = t(data)
            if data is None:
                return None
        return data
