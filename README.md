# SAT

## STAIR Captions

1. Dowload [datasets](http://captions.stair.center/) & unzip & place it in `./` (`./STAIR-captions`).
2. Download [COCO datasets](https://cocodataset.org/#download) & unzip & place it in `./` (`./train2014`).
3. Run `python create_stair_wmap.py > stair_word_map.json`
4. Run `python train.py -wm=stair_word_map.json`

- If facing `RuntimeError: unable to open shared memory object`, run `ulimit -n 65536` to increse the open files limit. `65536` is kinda arbitrary.

<!-- - Download [pretrained model](https://drive.google.com/drive/folders/189VY65I_n4RTpQnmLGj7IzVnOF6dmePC) -->
<!-- - `sh run.sh` -->