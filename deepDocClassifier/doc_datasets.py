import numpy as np
import pandas as pd
import torch
from PIL import Image
from pytorch_lightning import LightningDataModule
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.datasets import ImageFolder


class TobacoDataset(Dataset):
    def __init__(
            self,
            img_root_dir,
            num_training=20,
            ratio_validation=0.2,
            random_seed=42,
            split='training',
            transform=None
    ):
        super(TobacoDataset, self).__init__()
        self.img_root_dir = img_root_dir
        self.random_seed = random_seed
        self.split = split
        self.transform = transform
        ds = ImageFolder(self.img_root_dir)
        df = pd.DataFrame(ds.samples, columns=['img_path', 'label'])
        # add all samples to test split by default
        df['split'] = 'test'
        for i in ds.class_to_idx.values():
            indexes = np.array(df[df.label == i].index)
            # choose num_training number of index from above indexes
            np.random.seed(self.random_seed)
            indexes = np.random.choice(indexes, num_training, replace=False)
            # set selected num_training as training
            df.loc[indexes, 'split'] = 'training'
            # choose number of validation samples from above indexes as ratio_validation * num_training
            np.random.seed(self.random_seed)
            indexes = np.random.choice(indexes, int(num_training * ratio_validation), replace=False)
            df.loc[indexes, 'split'] = 'validation'

        self.df = df[df.split == self.split].reindex()
        self.classes = ds.classes
        self.class_to_idx = ds.class_to_idx

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        image_path = self.df.iloc[idx]['img_path']
        img = Image.open(image_path)
        label = self.df.iloc[idx]['label']
        img = transforms.ToTensor()(img)
        img = torch.cat((img, img, img), dim=0)
        if self.transform:
            img = self.transform(img)

        return img, label


class TobacoDataModule(LightningDataModule):
    def __init__(self,
                 img_root_dir,
                 num_training=20,
                 ratio_validation=0.2,
                 num_workers: int = 8,
                 batch_size: int = 32,
                 seed: int = 42,
                 shuffle: bool = True,
                 pin_memory: bool = False,
                 drop_last: bool = False,
                 ):
        super(TobacoDataModule, self).__init__()
        self.img_root_dir = img_root_dir
        self.num_training = num_training
        self.ratio_validation = ratio_validation
        self.num_workers = num_workers
        self.batch_size = batch_size
        self.seed = seed
        self.shuffle = shuffle
        self.pin_memory = pin_memory
        self.drop_last = drop_last
        self.num_classes = 10
        self.input_size = 227

        self.transform = transforms.Compose(
            [
                transforms.Resize((self.input_size, self.input_size)),
                # normalize using imagenet mean and variance
                transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),

            ]
        )

    def prepare_data(self) -> None:
        """
        Download data once. directly doenload using wget in colab, and skip here
        """
        pass

    def setup(self, stage=None):

        # Assign train/val datasets for use in dataloaders
        if stage == 'fit' or stage is None:
            self.ds_train = TobacoDataset(
                self.img_root_dir,
                self.num_training,
                self.ratio_validation,
                self.seed,
                split='training',
                transform=self.transform,
            )
            self.ds_val = TobacoDataset(
                self.img_root_dir,
                self.num_training,
                self.ratio_validation,
                self.seed,
                split='validation',
                transform=self.transform,
            )

        # Assign test dataset for use in dataloader(s)
        if stage == 'test' or stage is None:
            self.ds_test = TobacoDataset(
                self.img_root_dir,
                self.num_training,
                self.ratio_validation,
                self.seed,
                split='test',
                transform=self.transform,
            )

    def train_dataloader(self):
        return DataLoader(
            self.ds_train,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            drop_last=self.drop_last
        )

    def val_dataloader(self):
        return DataLoader(
            self.ds_val,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            drop_last=self.drop_last
        )

    def test_dataloader(self):
        return DataLoader(
            self.ds_test,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            drop_last=self.drop_last
        )
