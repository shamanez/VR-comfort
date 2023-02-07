
# Copyright The PyTorch Lightning team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Computer vision example on Transfer Learning. This computer vision example illustrates how one could fine-tune a
pre-trained network (by default, a ResNet50 is used) using pytorch-lightning. For the sake of this example, the
'cats and dogs dataset' (~60MB, see `DATA_URL` below) and the proposed network (denoted by `TransferLearningModel`,
see below) is trained for 15 epochs.
The training consists of three stages.
From epoch 0 to 4, the feature extractor (the pre-trained network) is frozen except
maybe for the BatchNorm layers (depending on whether `train_bn = True`). The BatchNorm
layers (if `train_bn = True`) and the parameters of the classifier are trained as a
single parameters group with lr = 1e-2.
From epoch 5 to 9, the last two layer groups of the pre-trained network are unfrozen
and added to the optimizer as a new parameter group with lr = 1e-4 (while lr = 1e-3
for the first parameter group in the optimizer).
Eventually, from epoch 10, all the remaining layer groups of the pre-trained network
are unfrozen and added to the optimizer as a third parameter group. From epoch 10,
the parameters of the pre-trained network are trained with lr = 1e-5 while those of
the classifier is trained with lr = 1e-4.
Note:
    See: https://pytorch.org/tutorials/beginner/transfer_learning_tutorial.html
To run:
    python computer_vision_fine_tuning.py fit
"""

import logging
from pathlib import Path
from typing import Union
import os

import torch
import torch.nn.functional as F
from torch import nn, optim
from torch.optim.lr_scheduler import MultiStepLR
from torch.optim.optimizer import Optimizer
from torch.utils.data import DataLoader
from torchmetrics import Accuracy
from torchvision import models
from torchvision.datasets import ImageFolder
from torchvision.datasets.utils import download_and_extract_archive

from  pytorchvideo import transforms
from  pytorchvideo import data as torch_video_data
from pytorchvideo.transforms import ApplyTransformToKey
from torchvision.transforms import Compose, Lambda


import pytorch_lightning as pl
from pl_examples import cli_lightning_logo
from pytorch_lightning import LightningDataModule
from pytorch_lightning.callbacks.finetuning import BaseFinetuning
from pytorch_lightning.utilities import rank_zero_info
from pytorch_lightning.utilities.cli import LightningCLI
from pytorch_lightning.callbacks import ModelCheckpoint

from flash.video import VideoClassificationData, VideoClassifier
#VideoClassifier.available_backbones()
#['c2d_r50', 'csn_r101', 'efficient_x3d_s', 'efficient_x3d_xs', 'i3d_r50', 'mvit_base_16', 'mvit_base_16x4', 'mvit_base_32x3', 'r2plus1d_r50', 'slow_r50', 'slow_r50_detection', 'slowfast_16x8_r101_50_50', 'slowfast_r101', 'slowfast_r50', 'slowfast_r50_detection', 'x3d_l', 'x3d_m', 'x3d_s', 'x3d_xs']



from pytorchvideo.transforms import (
    ApplyTransformToKey,
    ShortSideScale,
    UniformTemporalSubsample,
    UniformCropVideo
)


from torchvision.transforms._transforms_video import (
    CenterCropVideo,
    NormalizeVideo,
)


log = logging.getLogger(__name__)
DATA_URL = "https://pl-flash-data.s3.amazonaws.com/kinetics.zip"

#  --- Finetuning Callback ---


class MilestonesFinetuning(BaseFinetuning):
    def __init__(self, milestones: tuple = (1, 2), train_bn: bool = False):
        super().__init__()
        self.milestones = milestones
        self.train_bn = train_bn
     

    def freeze_before_training(self, pl_module: pl.LightningModule):
        print("not freezing anything")
        #self.freeze(modules=pl_module.feature_extractor, train_bn=self.train_bn)
    

    def finetune_function(self, pl_module: pl.LightningModule, epoch: int, optimizer: Optimizer, opt_idx: int):
        print("nothing")
        # if epoch == self.milestones[0]:
        #     # unfreeze 3 last layers
        #     self.unfreeze_and_add_param_group(
        #         modules=pl_module.feature_extractor[-3:], optimizer=optimizer, train_bn=self.train_bn
        #     )

        # elif epoch == self.milestones[1]:
        #     # unfreeze remaing layers
        #     self.unfreeze_and_add_param_group(
        #         modules=pl_module.feature_extractor[:-3], optimizer=optimizer, train_bn=self.train_bn
        #     )


# ----- Checkpoint Callback----


class CustomCheckpoint(ModelCheckpoint):
    def __init__(self,  monitor="val_loss",
                        dirpath="/hpc/gsir059/Shamane-Final/ckpt-score",
                        filename="sample-shamane-{epoch:02d}-{val_loss:.2f}",
                        save_top_k=3,
                        mode='min'):
        super().__init__(monitor=monitor,dirpath=dirpath,filename=filename,save_top_k=save_top_k,mode=mode)
     


#check https://pytorch.org/hub/facebookresearch_pytorchvideo_slowfast/
side_size = 256
mean = [0.45, 0.45, 0.45]
std = [0.225, 0.225, 0.225]
crop_size = 256
num_frames = 32
sampling_rate = 2
frames_per_second = 30
alpha = 4
class PackPathway(torch.nn.Module):
    """
    Transform for converting video frames as a list of tensors.
    """
    def __init__(self):
        super().__init__()

    def forward(self, frames: torch.Tensor):
        fast_pathway = frames
        # Perform temporal sampling from the fast pathway.
        slow_pathway = torch.index_select(
            frames,
            1,
            torch.linspace(
                0, frames.shape[1] - 1, frames.shape[1] // alpha
            ).long(),
        )
        frame_list = [slow_pathway, fast_pathway]
        return frame_list


class VrFacebookDataModule(LightningDataModule):
    def __init__(self, dl_path: Union[str, Path] = "data", num_workers: int =32, batch_size: int = 4):
        """VrFacebookDataModule.
        Args:
            dl_path: root directory where to download the data
            num_workers: number of CPU workers
            batch_size: number of sample in a batch
        """
        super().__init__()

        self._dl_path = dl_path
        self._num_workers = num_workers
        self._batch_size = batch_size

    def prepare_data(self):
        """Download images and prepare images datasets."""
        download_and_extract_archive(url=DATA_URL, download_root=self._dl_path, remove_finished=True)

    @property
    def data_path(self):
        return Path(self._dl_path).joinpath("VR-comfort")

    @property
    def train_transform(self):
        #transform_default=transforms.create_video_transform('train',video_key='video',num_samples=10,convert_to_float=False,horizontal_flip_prob=0)
        # transform_default=transforms.create_video_transform('train',num_samples=10,convert_to_float=False,horizontal_flip_prob=0)
        
        # transform =  ApplyTransformToKey(
        #                 key="video",
        #                 transform=Compose([transform_default,PackPathway()]))



        transform =  ApplyTransformToKey(
            key="video",
            transform=Compose(
                [
                    UniformTemporalSubsample(num_frames),
                    Lambda(lambda x: x/255.0),
                    NormalizeVideo(mean, std),
                    ShortSideScale(
                        size=side_size
                    ),
                    CenterCropVideo(crop_size),
                    PackPathway()
                ]
            ),
        )        
        return transform

    @property
    def valid_transform(self):
        # transform_default=transforms.create_video_transform('val',video_key='video',num_samples=10,convert_to_float=False,horizontal_flip_prob=0)
        # transform_default=transforms.create_video_transform('val',num_samples=10,convert_to_float=False,horizontal_flip_prob=0)
        # transform =  ApplyTransformToKey(
        #                 key="video",
        #                 transform=Compose([transform_default,PackPathway()]))

        transform =  ApplyTransformToKey(
            key="video",
            transform=Compose(
                [
                    UniformTemporalSubsample(num_frames),
                    Lambda(lambda x: x/255.0),
                    NormalizeVideo(mean, std),
                    ShortSideScale(
                        size=side_size
                    ),
                    CenterCropVideo(crop_size),
                    PackPathway()
                ]
            ),
        )
        return transform

    def create_dataset(self, root, transform):
        dataset= torch_video_data.Kinetics(data_path=root,
        clip_sampler=torch_video_data.make_clip_sampler("uniform",10),
        decode_audio=False,
        transform=transform
        )
        return dataset

    def __dataloader(self, train: bool):
        
        """Train/validation loaders."""
        if train:
            dataset = self.create_dataset(self.data_path.joinpath("train"), self.train_transform)
        else:
            dataset = self.create_dataset(self.data_path.joinpath("val"), self.valid_transform)
        return DataLoader(dataset=dataset, batch_size=self._batch_size, num_workers=self._num_workers)

    def train_dataloader(self):
        log.info("Training data loaded.")
        return self.__dataloader(train=True)

    def val_dataloader(self):
        log.info("Validation data loaded.")
        return self.__dataloader(train=False)


#  --- Pytorch-lightning module ---


class TransferLearningModel(pl.LightningModule):
    def __init__(
        self,
        backbone: str = "slowfast_r50",
        train_bn: bool = False,
        milestones: tuple = (2, 4),
        batch_size: int = 4,
        lr: float = 1e-4,
        lr_scheduler_gamma: float = 1e-1,
        num_workers: int = 32,
        **kwargs,
    ) -> None:
        """TransferLearningModel.
        Args:
            backbone: Name (as in ``torchvision.models``) of the feature extractor
            train_bn: Whether the BatchNorm layers should be trainable
            milestones: List of two epochs milestones
            lr: Initial learning rate
            lr_scheduler_gamma: Factor by which the learning rate is reduced at each milestone
        """
        super().__init__()
        self.backbone = backbone
        self.train_bn = train_bn
        self.milestones = milestones
        self.batch_size = batch_size
        self.lr = lr
        self.lr_scheduler_gamma = lr_scheduler_gamma
        self.num_workers = num_workers

        self.__build_model()

        self.train_acc = Accuracy()
        self.valid_acc = Accuracy()
        self.save_hyperparameters()

    def __build_model(self):
        """Define model layers & loss."""

        # 1. Load pre-trained network:
        backbone = torch.hub.load("facebookresearch/pytorchvideo", model=self.backbone, pretrained=True)
       
        # 2. Isolating the feature extractor:
        _layers = list(backbone.blocks[:-1])#list(backbone.children())[:-1]
        self.feature_extractor = nn.Sequential(*_layers)

        #get the slowfast projector
        projector=backbone.blocks[-1]
        projector.proj = torch.nn.Linear(2304, 1)

    
        # 3. Classifier:
        _fc_layers = [projector]#[nn.Linear(2304, 256), nn.ReLU(), nn.Linear(256, 32), nn.Linear(32, 1)]
        self.fc = nn.Sequential(*_fc_layers)

    

        # 3. Loss:
        self.loss_func = F.binary_cross_entropy_with_logits

    def forward(self, x):
        """Forward pass.
        Returns logits.
        """

 
        # 1. Feature extraction:
        x = self.feature_extractor(x)
      
    

        # 2. Classifier (returns logits):
        x = self.fc(x)
   
        return x

    def loss(self, logits, labels):
        return self.loss_func(input=logits, target=labels)

    def training_step(self, batch, batch_idx):
        # 1. Forward pass:
        x=batch['video']
        y=batch['label']

      
        y_logits = self.forward(x)
        y_scores = torch.sigmoid(y_logits)
        y_true = y.view((-1, 1)).type_as(y_logits)

      
        # 2. Compute loss
        train_loss = self.loss(y_logits,y_true)
     
      
        # 3. Compute accuracy:
        self.log("train_acc", self.train_acc(y_scores, y_true.int()), prog_bar=True)

        return train_loss

    def validation_step(self, batch, batch_idx):
        # 1. Forward pass:
   
        x=batch['video']
        y=batch['label'].view(-1,1)#.to(x[0].device)

        y_logits = self.forward(x)
        y_scores = torch.sigmoid(y_logits)
        y_true = y.view((-1, 1)).type_as(y_logits)

        # 2. Compute loss
        self.log("val_loss", self.loss(y_logits, y_true), prog_bar=True)

        # 3. Compute accuracy:
        self.log("val_acc", self.valid_acc(y_scores, y_true.int()), prog_bar=True)

    def configure_optimizers(self):
        parameters = list(self.parameters())
        trainable_parameters = list(filter(lambda p: p.requires_grad, parameters))
        rank_zero_info(
            f"The model will start training with only {len(trainable_parameters)} "
            f"trainable parameters out of {len(parameters)}."
        )
        optimizer = optim.Adam(trainable_parameters, lr=self.lr)
        scheduler = MultiStepLR(optimizer, milestones=self.milestones, gamma=self.lr_scheduler_gamma)
        return [optimizer], [scheduler]


class MyLightningCLI(LightningCLI):
    def add_arguments_to_parser(self, parser):
        parser.add_lightning_class_args(MilestonesFinetuning, "finetuning")
        parser.add_lightning_class_args(CustomCheckpoint, "ckpt")
        parser.link_arguments("data.batch_size", "model.batch_size")
        parser.link_arguments("finetuning.milestones", "model.milestones")
        parser.link_arguments("finetuning.train_bn", "model.train_bn")
        parser.set_defaults(
            {
                "trainer.max_epochs": 10,
                "trainer.enable_model_summary": False,
                "trainer.num_sanity_val_steps": 2,
            }
        )
  

def cli_main():

    MyLightningCLI(TransferLearningModel, VrFacebookDataModule, seed_everything_default=1234)


if __name__ == "__main__":
    cli_lightning_logo()
    cli_main()



#CUDA_VISIBLE_DEVICES=2,3 python finetune-video-cls.py fit --trainer.gpus 2 --trainer.accelerator ddp
#CUDA_VISIBLE_DEVICES=2,3,4,5 python finetune-video-cls.py fit --trainer.gpus 4 --trainer.accelerator ddp --trainer.val_check_interval 5000
