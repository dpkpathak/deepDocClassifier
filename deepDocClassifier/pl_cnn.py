"""
Implemetation of Kang, L., Kumar, J., Ye, P., Li, Y., & Doermann, D. (2014). Convolutional neural networks for document image classification. Proceedings - International Conference on Pattern Recognition, 3168–3172. https://doi.org/10.1109/ICPR.2014.546
"""
import os

import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
from PIL import Image
from pytorch_lightning import LightningModule, LightningDataModule
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from torch import nn
from torch.nn import functional as F
from torch.utils.data import Dataset, DataLoader
from torchmetrics.functional import accuracy
from torchvision import transforms
from torchvision.datasets import ImageFolder


class CNNDocPL(LightningModule):
    def __init__(self, num_classes: int = 10):
        super(CNNDocPL, self).__init__()

        # conv layer
        self.conv = nn.Sequential(
            # conv 1
            nn.Conv2d(1, 20, kernel_size=7, stride=2, padding=3),
            # nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=4, stride=2),

            # conv 2
            nn.Conv2d(20, 50, kernel_size=5, stride=2, padding=2),
            # nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=4, stride=2),
            nn.Flatten()

        )

        self.classifier = nn.Sequential(
            nn.Linear(50 * 8 * 8, 1000),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(1000, 1000),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(1000, num_classes),
        )

        # init param with Xavier method
        self.conv.apply(self.init_weights)
        self.classifier.apply(self.init_weights)

    @staticmethod
    def init_weights(layer):
        if type(layer) == nn.Linear or type(layer) == nn.Conv2d:
            nn.init.xavier_uniform_(layer.weight)

    def forward(self, x):
        x = self.conv(x)
        x = self.classifier(x)
        return F.log_softmax(x, dim=1)

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = F.nll_loss(logits, y)
        preds = torch.argmax(logits, dim=1)
        acc = accuracy(preds, y)

        self.log('train_loss', loss, prog_bar=True)
        self.log('train_acc', acc, prog_bar=True)
        return {
            'loss': loss,
            'acc': acc
        }

    def training_epoch_end(self, outputs):
        epoch_loss = torch.stack([x['loss'] for x in outputs]).mean()
        epoch_acc = torch.stack([x['acc'] for x in outputs]).mean()
        self.logger.experiment.add_scalar(f"train_epoch_loss", epoch_loss, self.current_epoch)
        self.logger.experiment.add_scalar(f"train_epoch_acc", epoch_acc, self.current_epoch)

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = F.nll_loss(logits, y)
        preds = torch.argmax(logits, dim=1)
        acc = accuracy(preds, y)
        self.log('val_loss', loss)

        return {
            'val_loss': loss,
            'acc': acc
        }

    def validation_epoch_end(self, outputs):
        epoch_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        epoch_acc = torch.stack([x['acc'] for x in outputs]).mean()
        self.logger.experiment.add_scalar(f"val_epoch_loss", epoch_loss, self.current_epoch)
        self.logger.experiment.add_scalar(f"val_epoch_acc", epoch_acc, self.current_epoch)

    def test_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        preds = torch.argmax(logits, axis=1)
        acc = accuracy(preds, y)
        return {
            'acc': acc,
            'y_real': y,
            'y_pred': preds
        }

    def test_epoch_end(self, outputs) -> None:
        epoch_acc = torch.stack([x['acc'] for x in outputs]).mean()
        y = torch.cat([x['y_real'] for x in outputs])
        pred = torch.cat([x['y_pred'] for x in outputs])
        results = torch.stack((y, pred), axis=1).cpu().numpy()
        results = pd.DataFrame(results, columns=['label', 'pred'])
        results.to_csv(
            os.path.join(self.logger.log_dir, "test_result.csv"),
            index=False
        )

        self.logger.experiment.add_scalar(f"test_epoch_acc", epoch_acc, self.current_epoch)
        self.log('test_epoch_acc', epoch_acc, logger=False, prog_bar=True, )

    def configure_optimizers(self):
        optimizer = torch.optim.SGD(
            params=self.parameters(),
            lr=0.001,
            momentum=0.9,
            weight_decay=0.0005,
        )
        return optimizer


class Tobacco4CNNDataset(Dataset):
    def __init__(
            self,
            img_root_dir,
            num_training=20,
            ratio_validation=0.2,
            random_seed=42,
            split='training',
            transform=None
    ):
        super(Tobacco4CNNDataset, self).__init__()
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
        img -= 0.5
        if self.transform:
            img = self.transform(img)

        return img, label


class Tobacco4CNNDataModule(LightningDataModule):
    def __init__(self,
                 img_root_dir,
                 num_training=20,
                 ratio_validation=0.2,
                 num_workers: int = 8,
                 batch_size: int = 10,
                 seed: int = 42,
                 shuffle: bool = True,
                 pin_memory: bool = False,
                 drop_last: bool = False,
                 ):
        super(Tobacco4CNNDataModule, self).__init__()
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
        self.input_size = 150

        self.transform = transforms.Compose(
            [
                transforms.Resize((self.input_size, self.input_size)),
                # normalize using imagenet mean and variance

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
            self.ds_train = Tobacco4CNNDataset(
                self.img_root_dir,
                self.num_training,
                self.ratio_validation,
                self.seed,
                split='training',
                transform=self.transform,
            )
            self.ds_val = Tobacco4CNNDataset(
                self.img_root_dir,
                self.num_training,
                self.ratio_validation,
                self.seed,
                split='validation',
                transform=self.transform,
            )

        # Assign test dataset for use in dataloader(s)
        if stage == 'test' or stage is None:
            self.ds_test = Tobacco4CNNDataset(
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


def main():
    # for deterministic training:
    pl.seed_everything(123)

    for num_training in [20, 40, 60, 80, 100]:
        # create data module
        dm = Tobacco4CNNDataModule(
            img_root_dir=r'C:\workspace\dfki-task\Tobacco3482-jpg',
            num_training=num_training,
            num_workers=0,
            batch_size=10,
        )

        # initialize model
        pl_module = CNNDocPL(
            num_classes=dm.num_classes,
        )

        model_checkpoint = ModelCheckpoint(save_last=True, save_top_k=1, monitor='val_loss')
        callbacks = [model_checkpoint, ]

        tb_logger = TensorBoardLogger(
            r'C:\workspace\dfki-task\deepDocClassifier\logs\CNNDoc',
            name='dataSize_' + str(num_training)
        )

        trainer = pl.Trainer(logger=tb_logger,
                             callbacks=callbacks,
                             gpus=1,
                             max_epochs=50,
                             precision=16,
                             progress_bar_refresh_rate=1,
                             log_every_n_steps=1,
                             )
        trainer.fit(
            pl_module,
            datamodule=dm
        )

        trainer.test()


if __name__ == '__main__':
    main()
