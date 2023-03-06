#!/bin/bash


cd ./PycharmProjects/classification/
python main.py --name fibrosis_noaug --batch_size 4 --image_size 1024 --save_dir ./ckpts --class_name uip,prob,indeter,other --augment none --filepath  ./CTMontage --gpu_ids 0
