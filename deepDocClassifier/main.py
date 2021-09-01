"""
This script runs trains the model with different configurations using YAML file
"""
import argparse
import json
from datetime import datetime
import gc

import numpy as np
import pytorch_lightning as pl
import torch
import yaml
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger

from deepDocClassifier.doc_datasets import TobacoDataModule
from deepDocClassifier.pl_alexnet import AlexNetPL

_ = np.__version__


def main():
    parser = argparse.ArgumentParser(description="Train AlexNet with different training data size")
    parser.add_argument('--config', '-c',
                        dest="filename",
                        metavar='FILE',
                        help='path to the config file',
                        default="../config/deepDocConfig.yaml")

    args = parser.parse_args()
    with open(args.filename, 'r') as file:
        try:
            config = yaml.safe_load(file)
        except yaml.YAMLError as exc:
            print(exc)
            exit(-1)

    # for deterministic training:
    pl.seed_everything(config['experiment']['manual_seed'])
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    # load different dataset size for training data
    for num_training in config['training_dataset_sizes']:
        # create data module
        dm = TobacoDataModule(
            img_root_dir=config['dataset_params']['img_root_dir'],
            num_training=num_training,
            num_workers=config['dataset_params']['num_workers'],
            batch_size=config['dataset_params']['batch_size'],
        )

        # initialize model
        pl_module = AlexNetPL(
            num_classes=dm.num_classes,
            version=config['model_param']['version']
        )

        model_checkpoint = ModelCheckpoint(save_last=True, save_top_k=1, monitor='val_loss')
        callbacks = [model_checkpoint, ]

        tb_logger = TensorBoardLogger(
            config['logging_params']['log_dir'] + '-' + timestamp,
            name='' + 'dataSize_' + str(num_training)
        )

        # save parameters of the config file to the tensorboard
        tb_logger.experiment.add_text("configurations_used", pretty_json(config))

        trainer = pl.Trainer(logger=tb_logger,
                             callbacks=callbacks,
                             **config['trainer_params'],
                             )

        trainer.fit(
            pl_module,
            datamodule=dm
        )

        trainer.test()

        # free gpu cache
        del pl_module
        del trainer
        gc.collect()

        torch.cuda.empty_cache()


def pretty_json(hp):
    """
    Use to log test in tensorboard
    :param hp:
    :return:
    """
    json_hp = json.dumps(hp, indent=2)
    return "".join("\t" + line for line in json_hp.splitlines(True))


if __name__ == '__main__':
    main()
