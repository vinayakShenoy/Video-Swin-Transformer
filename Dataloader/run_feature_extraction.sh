#!/bin/bash

raw_frames_folder="../data/jigsaw/suturing_list_rawframes"
for entry in "${raw_frames_folder}/";
do
  echo $entry
  #sed -i "10s/.*/ann_file_train = $entry" ../configs/recognition/swin/swin_tiny_patch244_window877_kinetics400_1k.py
  #echo "$entry"
done


