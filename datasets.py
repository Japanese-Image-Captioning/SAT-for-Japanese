import torch
from torch.utils.data import Dataset
import h5py
import json
import os
import torchvision
import cv2
import colored_traceback.always
from tqdm import tqdm
from PIL import Image

class StairCaptionDataset(Dataset):
    def __init__(self, data_folder, data_name, split, transform=None):
        print(f"Preprocess for {split} ... ")
        dataset = None
        if split == "train":
            with open("STAIR-captions/stair_captions_v1.2_train_tokenized.json", 'r') as f:
                dataset = json.load(f)
        else:
            with open("STAIR-captions/stair_captions_v1.2_val_tokenized.json", 'r') as f:
                dataset = json.load(f)

        with open("stair_word_map.json", 'r') as f:
            wmap = json.load(f)
        
        labels_dict = {} # labels[image_id] = captions
        for cmeta in tqdm(dataset["annotations"]):
            image_id = cmeta["image_id"]
            caption = cmeta["tokenized_caption"].split(' ')
            caplen = len(caption)
            tokens = self.tokenize(caption, wmap)
            labels_dict.setdefault(image_id, [])
            labels_dict[image_id].append((tokens,caplen))
        

        labels = []
        if split == "train":
            for image_id, captions in tqdm(labels_dict.items()):
                for caption, caplen in captions:
                    labels.append((image_id,caption, caplen)) # image_id, caption

        else:
            img_idxs = list(labels_dict.keys())
            half_size = len(img_idxs) // 2
            img_idxs = img_idxs[:half_size] if split == "val" else img_idxs[half_size:] # COCOのvalを半分に分割
            for image_id in tqdm(img_idxs):
                for caption, caplen in labels_dict[image_id]:
                    labels.append((image_id,caption, caplen)) # image_id, caption
            

        self.labels = labels
        self.labels_dict = labels_dict
        self.wmap = wmap
        self.split = split
        self.transform = transform

    def __getitem__(self, i):
        image_id, caption, caplen = self.labels[i]
        img = self.get_coco_image(image_id, self.split)
        if img.shape[0] == 1:
            img = torch.cat([img]*3, dim=0)

        assert img.shape[0] == 3, f"img.shape == {img.shape}"
        if self.transform is not None:
            img = self.transform(img)

        caplen = torch.LongTensor([caplen + 2])
        caption = torch.LongTensor(caption)

        if self.split == 'train':
            return img, caption, image_id, caplen
        else:
            all_captions = torch.LongTensor([cap for cap, _ in self.labels_dict[image_id]])
            # assert len(all_captions) == 5, f"image_id: {image_id} (len={len(all_captions)})"
            return img, caption, image_id, caplen, all_captions[:5]


    def __len__(self):
        return len(self.labels)

    def tokenize(self, target, wmap):
        maxlen = 100
        x = [wmap['<start>']] + [wmap[t] if t in wmap else wmap['<unk>'] for t in target] + [wmap['<end>']]
        assert len(x) <= maxlen, f'length={len(x)} over {maxlen}'

        return x + [wmap['<pad>']] * (maxlen - len(x))

    def get_coco_image(self, image_id, split):
        if split == "test":
            split = "val" # COCOのvalを使う
        
        L = len("000000490055")
        prefix = "0"*(L - len(str(image_id)))
        path = f"{split}2014/COCO_{split}2014_{prefix}{image_id}.jpg"
        resized_path = f"{split}2014/resized_COCO_{split}2014_{prefix}{image_id}.jpg"

        if os.path.exists(resized_path):
            path = resized_path
        else:
            img = cv2.imread(path)
            img = cv2.resize(img,(256,256))
            cv2.imwrite(resized_path,img)
            path = resized_path

        img_pil = Image.open(path)
        img_tensor = torchvision.transforms.functional.to_tensor(img_pil)
        return img_tensor


class COCOCaptionDataset(StairCaptionDataset):
    def __init__(self, data_folder, data_name, split, transform=None):
        super().__init__(data_folder, data_name, split, transform)
        stair_labels_dict = self.labels_dict

        dataset = None
        if split == "train":
            with open("annotations/captions_train2014.json", 'r') as f:
                dataset = json.load(f)
        else:
            with open("annotations/captions_val2014.json", 'r') as f:
                dataset = json.load(f)

        with open("coco_word_map.json", 'r') as f:
            wmap = json.load(f)
        
        labels_dict = {} # labels[image_id] = captions
        for cmeta in tqdm(dataset["annotations"]):
            image_id = cmeta["image_id"]
            if image_id not in stair_labels_dict: continue # only images contained in STAIR
            caption = [c for c in cmeta["caption"].split(' ') if len(c) > 0]
            caplen = len(caption)
            tokens = self.tokenize(caption, wmap)
            labels_dict.setdefault(image_id, [])
            labels_dict[image_id].append((tokens,caplen))
        
        labels = []
        if split == "train":
            for image_id, captions in tqdm(labels_dict.items()):
                for caption, caplen in captions:
                    labels.append((image_id,caption, caplen)) # image_id, caption

        else:
            img_idxs = list(labels_dict.keys())
            half_size = len(img_idxs) // 2
            img_idxs = img_idxs[:half_size] if split == "val" else img_idxs[half_size:] # COCOのvalを半分に分割
            for image_id in tqdm(img_idxs):
                for caption, caplen in labels_dict[image_id]:
                    labels.append((image_id,caption, caplen)) # image_id, caption
            

        self.labels = labels
        self.labels_dict = labels_dict
        self.wmap = wmap



class CaptionDataset(Dataset):
    """
    A PyTorch Dataset class to be used in a PyTorch DataLoader to create batches.
    """

    def __init__(self, data_folder, data_name, split, transform=None):
        """
        :param data_folder: folder where data files are stored
        :param data_name: base name of processed datasets
        :param split: split, one of 'TRAIN', 'VAL', or 'TEST'
        :param transform: image transform pipeline
        """
        self.split = split
        assert self.split in {'TRAIN', 'VAL', 'TEST'}

        # Open hdf5 file where images are stored
        self.h = h5py.File(os.path.join(data_folder, self.split + '_IMAGES_' + data_name + '.hdf5'), 'r')
        self.imgs = self.h['images']

        # Captions per image
        self.cpi = self.h.attrs['captions_per_image']

        # Load encoded captions (completely into memory)
        with open(os.path.join(data_folder, self.split + '_CAPTIONS_' + data_name + '.json'), 'r') as j:
            self.captions = json.load(j)

        # Load caption lengths (completely into memory)
        with open(os.path.join(data_folder, self.split + '_CAPLENS_' + data_name + '.json'), 'r') as j:
            self.caplens = json.load(j)

        # PyTorch transformation pipeline for the image (normalizing, etc.)
        self.transform = transform

        # Total number of datapoints
        self.dataset_size = len(self.captions)

    def __getitem__(self, i):
        # Remember, the Nth caption corresponds to the (N // captions_per_image)th image
        img = torch.FloatTensor(self.imgs[i // self.cpi] / 255.)
        if self.transform is not None:
            img = self.transform(img)

        caption = torch.LongTensor(self.captions[i])

        caplen = torch.LongTensor([self.caplens[i]])

        if self.split == 'train':
            return img, caption, caplen
        else:
            # For validation of testing, also return all 'captions_per_image' captions to find BLEU-4 score
            all_captions = torch.LongTensor(
                self.captions[((i // self.cpi) * self.cpi):(((i // self.cpi) * self.cpi) + self.cpi)])
            return img, caption, caplen, all_captions

    def __len__(self):
        return self.dataset_size
