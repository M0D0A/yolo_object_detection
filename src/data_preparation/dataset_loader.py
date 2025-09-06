import torch
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import tv_tensors
import xml.etree.ElementTree as ET
from pathlib import Path
from .augmentation import Augmentation
from ..utils.conver_box import xyxy2xywh
from PIL import Image


class PascalVOC:
    def __init__(self, path):
        self.path = Path(path)
        self.anns_path = self.path.joinpath("Annotations")
        self.imgs_path = self.path.joinpath("JPEGImages")
        self.img_formats = (".bmp", ".dng", ".jpeg", ".jpg", ".mpo", ".png", ".tif", ".tiff", ".webp", ".pfm")
        
        classes = [
            'person', 'bird', 'cat', 'cow', 'dog', 'horse', 'sheep',
            'aeroplane', 'bicycle', 'boat', 'bus', 'car', 'motorbike', 'train',
            'bottle', 'chair', 'diningtable', 'pottedplant', 'sofa', 'tvmonitor'
        ]

        self.classes = sorted(classes)
        self.cls_to_idx = {i: cls for i, cls in enumerate(self.classes)}
        self.idx_to_cls = {cls: i for i, cls in enumerate(self.classes)}


    def preparation_data(self):
        '''
        new_ann = {
            'img_path': str, путь до изображения,
            'img_name': str, имя файла с изображением,
            'img_size': [int, int], размер изображения,
            'depth': int, кол-во каналов,
            'objs_ann': [
                {
                    'label': int, класс объекта,
                    'bnbox': [int, int, int, int], параметры ограничивающей рамки (x1y1x2y2)
                },
                ...
                { label и bnbox N-го объекта }
            ]
        }

        data = [img_data_1, ... img_data_N]
        '''
        data = []
        for img_path in self.imgs_path.iterdir():
            img_format = img_path.suffix in self.img_formats
            ann_path = self.anns_path.joinpath(f"{img_path.name.split('.')[0]}.xml")
            exists = ann_path.exists()

            if not(img_format) or not(exists):
                continue
            
            new_ann = {}
            ann = ET.parse(ann_path)

            new_ann["img_path"] = img_path
            new_ann["img_name"] = img_path.name
            new_ann["img_size"] = (
                int(ann.find("size").find("width").text),
                int(ann.find("size").find("height").text)
            )
            new_ann["depth"] = int(ann.find("size").find("depth").text)

            objs_ann = []
            for obj in ann.findall("object"):
                obj_ann = {}
                obj_ann["label"] = self.idx_to_cls.get(obj.find("name").text)
                obj_ann["bnbox"] = (
                    int(float(obj.find("bndbox").find("xmin").text)) - 1,
                    int(float(obj.find("bndbox").find("ymin").text)) - 1,
                    int(float(obj.find("bndbox").find("xmax").text)) - 1,
                    int(float(obj.find("bndbox").find("ymax").text)) - 1,
                )
                objs_ann.append(obj_ann)

            new_ann["objs_ann"] = objs_ann
            data.append(new_ann)

        return data


class DatasetPascalVOC(Dataset):
    def __init__(self, data, transforms=None, mode=None):
        self.data = data
        self.transforms = transforms if transforms else Augmentation(mode)
        self.S = 7
        self.B = 2
        self.C = 20

        classes = [
            'person', 'bird', 'cat', 'cow', 'dog', 'horse', 'sheep',
            'aeroplane', 'bicycle', 'boat', 'bus', 'car', 'motorbike', 'train',
            'bottle', 'chair', 'diningtable', 'pottedplant', 'sofa', 'tvmonitor'
        ]

        self.classes = sorted(classes)
        self.cls_to_idx = {i: cls for i, cls in enumerate(self.classes)}
        self.idx_to_cls = {cls: i for i, cls in enumerate(self.classes)}


    def __len__(self):
        return len(self.data)


    def __getitem__(self, index):
        data = self.data[index]
        img = Image.open(data["img_path"]).convert("RGB")

        W, H = data["img_size"]
        bnboxes = [objs_ann["bnbox"] for objs_ann in data["objs_ann"]]
        bnboxes = tv_tensors.BoundingBoxes(
            bnboxes,
            format="XYXY",
            canvas_size=(H, W)
        )
        labels = [objs_ann["label"] for objs_ann in data["objs_ann"]]

        target_bl = {
            "boxes": bnboxes,
            "labels": torch.tensor(labels)
        }

        img, target_bl = self.transforms(img, target_bl)
        bnboxes = target_bl["boxes"]
        labels = target_bl["labels"]

        whwh = torch.tensor([448,448,448,448])
        bnboxes_xyxy = bnboxes / whwh

        bnboxes_classes = torch.cat([bnboxes_xyxy, labels.unsqueeze(dim=1)], dim=1)

        bnboxes_xywh = xyxy2xywh(bnboxes_xyxy)

        target = torch.zeros(self.S, self.S, 5 * self.B + self.C)

        n_l = len(labels)
        cls_labels = torch.zeros((n_l, self.C))
        cls_labels[[*range(n_l)], labels] = 1
        conf = torch.ones((n_l, 1))

        wh_cell = torch.tensor([1 / self.S, 1 / self.S])
        idx = torch.floor(bnboxes_xywh[:, :2] / wh_cell)
        ij = torch.tensor_split(idx.long(), 2, dim=1)

        bnboxes_xywh[:,:2] = (bnboxes_xywh[:,:2] % wh_cell) / wh_cell
        bnboxes_vectors = torch.cat([bnboxes_xywh, conf, bnboxes_xywh, conf, cls_labels], dim=1)
        target[ij[1], ij[0]] = bnboxes_vectors[:, None, :]
        
        return img, target, bnboxes_classes


    @staticmethod
    def collate_fn(batch):
        data = list(zip(*batch))
        imgs = torch.stack(data[0])
        targets = torch.stack(data[1])
        box_class = data[2]
        return imgs, targets, box_class


def create_loader(
        path,
        batch_size = 16,
        train_shuffle = True,
        split = [0.8, 0.2],
        num_workers = 1,
    ):

    data = PascalVOC(path).preparation_data()
    train_data, val_data = random_split(data, split)

    train_dataset = DatasetPascalVOC(train_data, mode="train")
    val_dataset = DatasetPascalVOC(val_data, mode="test")

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=train_shuffle,
        num_workers=num_workers,
        collate_fn=train_dataset.collate_fn
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=train_dataset.collate_fn
    )
    
    return train_loader, val_loader
