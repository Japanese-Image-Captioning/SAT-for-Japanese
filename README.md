# SAT for Japanese

<img width="827" alt="Screen Shot 2022-10-05 at 0 03 01" src="https://user-images.githubusercontent.com/51681991/193856884-89eef39e-6710-4f38-9b77-86f6d706f49a.png">


- [Show, Attend and Tell: Neural Image Caption Generation with Visual Attention](https://arxiv.org/abs/1502.03044)
- This repository is based on [the repo](https://github.com/sgrvinod/a-PyTorch-Tutorial-to-Image-Captioning).
- This code uses [STAIR Captions](http://captions.stair.center/) for training Japanese image captioning model.
  - So you should download STAIR Captions and [COCO datasets](https://cocodataset.org/#download)
 
## Instructions

1. Dowload [datasets](http://captions.stair.center/) & unzip & place it in `./` (`./STAIR-captions`).
2. Download [COCO datasets](https://cocodataset.org/#download) & unzip & place it in `./` (`./train2014`, `./val2014`).
3. Run `python create_stair_wmap.py > stair_word_map.json`

- You can download `stair_word_map.json`, checkpoints, and `coco_word_map.json` from [here](https://github.com/Japanese-Image-Captioning/SAT-for-Japanese/releases/tag/v1.0.0)!

### Train 

```
python train.py -wm=stair_word_map.json
```

- Alias : `sh train.sh`

- The `-en` option allows you to train on COCO datasets with the same data set partitioning method as STAIR Captions

```
python train.py -wm=XXX.json --en
```

- If facing `RuntimeError: unable to open shared memory object`, run `ulimit -n 65536` to increse the open files limit. `65536` is kinda arbitrary.

### Generate Caption

```
  python caption.py --model=stair_checkpoints/best.pth.tar -wm=stair_word_map.json --img=<any image>
```

### Examples

![0](https://user-images.githubusercontent.com/51681991/193860716-6528a04f-7ff0-4d54-9ee2-4f9abe34f063.jpg)

output: 警察 の バイク が 展示 さ れ て いる

<br>

![2](https://user-images.githubusercontent.com/51681991/193860839-65be962e-6a40-4a1e-861b-c6f926fb8c05.jpg)

output: トイレ の 便座 が 上がっ て いる

<br>

![5](https://user-images.githubusercontent.com/51681991/193859506-541b8096-075c-49de-a7d3-155933da2df0.jpg)

output: テーブル の 上 に 料理 が 並ん で いる

<br>

![3](https://user-images.githubusercontent.com/51681991/193861051-ba91b980-e1f0-49e7-a77e-855545c58519.jpg)

output: 時計 塔 の 上 に 時計 が つい て いる

<br>

![6](https://user-images.githubusercontent.com/51681991/193859633-36428b6c-9d06-478e-ba72-94322bb3f041.jpg)

output: 白い 服 を 着 た 男性 が 食事 を し て いる

<br>

![5](https://user-images.githubusercontent.com/51681991/193860390-e3e63d23-a04a-4892-aec6-9b2085b40158.jpg)

output: 男性 が キッチン で 料理 を し て いる

<br>

## Others

### Licence

- a-PyTorch-Tutorial-to-Image-Captioning
  - https://github.com/sgrvinod/a-PyTorch-Tutorial-to-Image-Captioning

This work is licensed under the MIT License. To view a copy of this license, see LICENSE.



