#!/bin/bash

rawframes_list_TCN="../data/jigsaw/suturing_TCN_annotations"

for entry in "$rawframes_list_TCN"/*;do
    python feature_extractor.py \
  --model_checkpoint ../models/swin_base_patch244_window877_kinetics600_22k.pth \
  --model_config ../configs/recognition/swin/swin_base_patch244_window877_kinetics600_22k.py \
  --batch_size 1 \
  --num_workers 4 \
  --videos_per_gpu 8 \
  --annotation_file $entry \
  --data_prefix ../data/jigsaw/rawframes_train_TCN_modified/ \
  --transcriptions_dir ../data/jigsaw/Suturing/transcriptions \
  --features_output_dir features_dir/
  echo "$entry"
done
