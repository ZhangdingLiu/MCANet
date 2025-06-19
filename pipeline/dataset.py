import json
from torch.utils.data import Dataset
from PIL import Image
from torchvision.transforms import transforms
import torch
from torchvision.transforms import RandAugment



class DataSet(Dataset):
    def __init__(self,   #usage:  train_dataset = DataSet(train_file, args.train_aug, args.img_size, args.dataset)
                ann_files,
                augs,
                img_size,
                dataset,
                ):
        self.dataset = dataset
        self.ann_files = ann_files
        self.augment = self.augs_function(augs, img_size)
        # self.transform = transforms.Compose(
        #     [
        #         transforms.ToTensor(),
        #         transforms.Normalize(mean=[0, 0, 0], std=[1, 1, 1])
        #     ]
        #     # In this paper, we normalize the image data to [0, 1]
        #     # You can also use the so called 'ImageNet' Normalization method
        # )
        self.transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ]
        )
        self.anns = []
        self.load_anns()
        print(self.augment)


    def augs_function(self, augs, img_size):            
        t = []
        if 'randomflip' in augs:
            t.append(transforms.RandomHorizontalFlip())
        if 'ColorJitter' in augs:
            t.append(transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0))
        if 'resizedcrop' in augs:
            t.append(transforms.RandomResizedCrop(img_size, scale=(0.7, 1.0)))
        if 'RandAugment' in augs:
            t.append(RandAugment())

        t.append(transforms.Resize((img_size, img_size)))

        return transforms.Compose(t)
    
    def load_anns(self):
        self.anns = []
        for ann_file in self.ann_files:
            json_data = json.load(open(ann_file, "r"))
            self.anns += json_data

    def __len__(self):
        return len(self.anns)

    def __getitem__(self, idx):
        idx = idx % len(self)
        ann = self.anns[idx]
        img = Image.open(ann["img_path"]).convert("RGB")

        if self.dataset == "wider":
            x, y, w, h = ann['bbox']
            img_area = img.crop([x, y, x+w, y+h])
            img_area = self.augment(img_area)
            img_area = self.transform(img_area)
            message = {
                "img_path": ann['img_path'],
                "target": torch.Tensor(ann['target']),
                "img": img_area
            }
        else: # voc and coco
            img = self.augment(img)
            img = self.transform(img)
            message = {
                "img_path": ann["img_path"],
                "target": torch.Tensor(ann["target"]),
                "img": img
            }

        return message
