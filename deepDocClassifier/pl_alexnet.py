import os

import pandas as pd
import pytorch_lightning as pl
import torch
from torch import nn
from torch.nn import functional as F
from torchmetrics.functional import accuracy
from torchvision.models import alexnet

from deepDocClassifier.alexnet import AlexNet


class AlexNetPL(pl.LightningModule):
    """

    """

    def __init__(self, num_classes: int = 10, version: str = 'torchvision'):
        """

        :param num_classes:
        :param version: can be used to create different version of AlexNet.
            'pytorch':`"One weird trick..." <https://arxiv.org/abs/1404.5997>`_ paper.
            'original': `"Imagenet classification with deep convolutional neural networks..." <https://papers.nips.cc/paper/2012/file/c399862d3b9d6b76c8436e924a68c45b-Paper.pdf>` paper
        """
        super(AlexNetPL, self).__init__()
        self.version = version
        self.num_classes = num_classes

        self.model = self.init_model()

    def init_model(self, ):
        if self.version == 'original':
            model = AlexNet(num_classes=self.num_classes)
            return model
        elif self.version == 'torchvision':
            model = alexnet(pretrained=True, )
            model.classifier = nn.Sequential(
                *list(model.classifier)[:-1],
                nn.Linear(4096, 10),
            )
            return model

    def forward(self, x):
        x = self.model(x)
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
        y = torch.stack([x['y_real'] for x in outputs]).reshape((len(outputs) * 10, 1))
        pred = torch.stack([x['y_pred'] for x in outputs]).reshape((len(outputs) * 10, 1))
        results = torch.cat((y, pred), dim=1).cpu().numpy()
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
            lr=0.0001,
            momentum=0.9,
            weight_decay=0.0005,
        )
        return optimizer
