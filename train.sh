#!/bin/sh

cnt=$#
ja=1
if [ $cnt -gt 0 ]; then
    if [ $1 = "en" ]; then
        ja=0
    fi
fi

if [ ! -e stair_word_map.json ]; then
    python create_stair_wmap.py > stair_word_map.json
fi

if [ ! -e coco_word_map.json ]; then
    python create_stair_wmap.py --en > coco_word_map.json
fi

ulimit -n 65536

if [ $ja -eq 1 ]; then
    python train.py -wm=stair_word_map.json --wandb
else
    python train.py -wm=coco_word_map.json --en --wandb
fi