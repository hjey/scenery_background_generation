import os
from glob import glob
import cv2
import numpy as np
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torch.utils.data import Dataset, DataLoader


# 1. train -> ADE20K (with augmentation)
# 2. predict -> Landscape
def get_dataloader(batch_size, mode='train', transform=list()):
    assert mode == 'train' or (mode == 'predict' and batch_size == 1)

    n_cpu = os.cpu_count()
    default_transform = [
        A.Resize(512, 512),
        ToTensorV2(transpose_mask=True)
    ]
    valid_transform = A.Compose(default_transform, is_check_shapes=False)

    if mode == 'train':
        transform.extend(default_transform)
        train_transform = A.Compose(transform, is_check_shapes=False)
        train_dataset = ADE20KOutdoorDataset('train', train_transform)
        valid_dataset = ADE20KOutdoorDataset('valid', valid_transform)
        test_dataset  = ADE20KOutdoorDataset('test',  valid_transform)
        train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,  num_workers=n_cpu)
        valid_dataloader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False, num_workers=n_cpu)

    elif mode == 'predict':
        train_dataset, train_dataloader = list(), None
        valid_dataset, valid_dataloader = list(), None
        test_dataset = LandScapeDataset(valid_transform)

    test_dataloader  = DataLoader(test_dataset,  batch_size=batch_size, shuffle=False, num_workers=n_cpu)
    print(f"Train size: {len(train_dataset)}")
    print(f"Valid size: {len(valid_dataset)}")
    print(f"Test size:  {len(test_dataset)}")
    return train_dataloader, valid_dataloader, test_dataloader



# Train / Valid / Test Dataset
class ADE20KOutdoorDataset(Dataset):
    def __init__(self, mode='train', transforms=None):
        image_paths = sorted(glob('/home/work/Backscape/train_data/images/training/*.jpg'))
        label_paths = sorted(glob('/home/work/Backscape/train_data/annotations-custom/training/*.png'))

        if   mode == 'train':
            self.image_paths = image_paths[:int(len(image_paths) * 0.8)]
            self.label_paths = label_paths[:int(len(image_paths) * 0.8)]
        elif mode == 'valid':
            self.image_paths = image_paths[int(len(image_paths) * 0.8):int(len(image_paths) * 0.95)]
            self.label_paths = label_paths[int(len(image_paths) * 0.8):int(len(image_paths) * 0.95)]
        elif mode == 'test':
            self.image_paths = image_paths[int(len(image_paths) * 0.95):]
            self.label_paths = label_paths[int(len(image_paths) * 0.95):]
        self.transforms = transforms

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path, label_path = self.image_paths[idx], self.label_paths[idx]

        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        label = cv2.imread(label_path, cv2.IMREAD_GRAYSCALE)[..., np.newaxis]
        if self.transforms is not None:
            transformed = self.transforms(image=image, mask=label)
            image = transformed["image"]
            label = transformed["mask"]
        return {"image": image, "label": label}
    

# Predict Dataset
class LandScapeDataset(Dataset):
    def __init__(self, transforms=None):
        self.image_paths = sorted(glob('/home/work/Backscape/test_data/images/*.jpg'))
        self.transforms = transforms

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]

        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        if self.transforms is not None:
            transformed = self.transforms(image=image)
            image = transformed["image"]
        return {"image": image}
