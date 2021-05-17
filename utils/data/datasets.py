from torch.utils.data import Dataset, DataLoader
from torchvision.datasets import ImageFolder
from PIL import Image, ImageFile
from pathlib import Path
import numpy as np
import os
import pickle
import torch
import torch.utils.data.distributed
ImageFile.LOAD_TRUNCATED_IMAGES = True


class ImageListDataset(Dataset):
    def __init__(self, img_list, folder='./', transform=None):
        self.folder = folder
        self.transform = transform
        self.path_list = [line.split(' ')[0] for line in open(img_list)]
        self.label_list = [int(line.split(' ')[1].strip('\n')) for line in open(img_list)]

    def __len__(self):
        return len(self.label_list)

    def __getitem__(self, idx):
        image = self.transform(Image.open(os.path.join(self.folder, self.path_list[idx])).convert('RGB'))
        return image, self.label_list[idx]

    @property
    def class_num(self):
        return np.unique(self.label_list).shape[0]


def get_train_dataset(train_transform, img_folder, img_list=None):
    if img_list is None:
        ds = ImageFolder(img_folder, train_transform)
        class_num = ds[-1][1] + 1
    else:
        ds = ImageListDataset(img_list=img_list, folder=img_folder, transform=train_transform)
        class_num = ds.class_num
    return ds, class_num


def img_list_dataloader(traindir, img_list, train_transform, distributed=False,
                        batch_size=64, num_workers=32, pin_memory=True):
    train_set, class_num = get_train_dataset(train_transform, traindir, img_list)
    if distributed:
        train_set_index = range(len(train_set))
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_set_index)
    else:
        train_sampler = None
    loader = DataLoader(train_set, batch_size=batch_size, shuffle=(train_sampler is None),
                        pin_memory=pin_memory, num_workers=num_workers, sampler=train_sampler)
    return loader, class_num, train_sampler
