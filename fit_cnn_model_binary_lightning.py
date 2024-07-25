import pickle
import argparse

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split

import lightning as L
from lightning.pytorch.utilities.types import TRAIN_DATALOADERS

from torchmetrics import Accuracy, Recall, Precision, ConfusionMatrix

from fit_cnn_model_binary import ComPairNet, TestNet1, VoxelDataset
from models import gen_testnet1

class EventTypeIdentification(L.LightningModule):
    def __init__(self, input_shape=(110, 110, 48)):
        super().__init__()

        # Metrics
        self.train_acc = Accuracy(task="binary")
        self.train_recall = Recall(task="binary")
        self.train_recall0 = Recall(task="binary")
        self.train_prec = Precision(task="binary")
        self.train_prec0 = Precision(task="binary")
        self.train_cm = ConfusionMatrix(task="binary")

        self.val_acc = Accuracy(task="binary")
        self.val_recall = Recall(task="binary")
        self.val_prec = Precision(task="binary")
        self.val_recall0 = Recall(task="binary")
        self.val_prec0 = Precision(task="binary")
        self.val_cm = ConfusionMatrix(task="binary")

        # Model
        
        # Convolutional Layers
        # self.conv1 = torch.nn.Conv3d(in_channels=1, out_channels=32, kernel_size=5, stride=2)
        # self.conv2 = torch.nn.Conv3d(in_channels=32, out_channels=32, kernel_size=3)
        # self.conv3 = torch.nn.Conv3d(in_channels=32, out_channels=32, kernel_size=5)
        # self.conv4 = torch.nn.Conv3d(in_channels=128, out_channels=128, kernel_size=5)
        
        # self.relu = torch.nn.ReLU()
        # self.leakyrelu0p01 = torch.nn.LeakyReLU(0.01)
        # self.leakyrelu0p1 = torch.nn.LeakyReLU(0.1)

        # self.dropout0p5 = torch.nn.Dropout(0.5)
        # self.dropout0p4 = torch.nn.Dropout(0.4)
        # self.dropout0p2 = torch.nn.Dropout(0.2)
        # self.dropout3d0p5 = torch.nn.Dropout3d(0.5)

        # self.conv_list = [self.conv1, self.relu, 
                        #   self.conv2, self.relu, self.dropout3d0p5]
                        #   self.conv3, self.relu, self.dropout3d0p5]
                        #   self.conv4, self.relu, self.dropout3d0p5]
        
        # x = self.cnnpart(torch.autograd.Variable(
            # torch.rand((1, 1) + input_shape)))
        # first_fc_in_features = x.size()[1:].numel()
        # print("Number of features:", first_fc_in_features)
        
        # self.flatten = torch.nn.Flatten()
        # self.lfc1 = torch.nn.Linear(first_fc_in_features, 128)
        # self.lfc2 = torch.nn.Linear(128, 1)

        # self.layer_list = gen_testnet1(input_shape=input_shape)
        # print("AAA:", self.layer_list)

        self.model = gen_testnet1(input_shape=input_shape)

    '''
    def cnnpart(self, x):
        x = self.conv1(x)
        x = self.leakyrelu0p01(x)
        x = self.dropout0p4(x)
        x = self.conv2(x)
        x = self.leakyrelu0p01(x)
        x = self.dropout0p4(x)
        x = self.conv3(x)
        x = self.leakyrelu0p01(x)
        x = self.dropout0p4(x)

        return x
    
    def fullyconnectedpart(self, x):
        x = self.flatten(x)
        x = self.lfc1(x)
        x = self.leakyrelu0p01(x)
        x = self.dropout0p4(x)
        x = self.lfc2(x)

        return x

    def forward(self, x):
        x = self.cnnpart(x)
        x = self.fullyconnectedpart(x)
        return x
    '''

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        y = y.view_as(logits)
        loss = nn.functional.binary_cross_entropy_with_logits(logits, y)
        
        self.train_acc.update(logits, y)
        self.train_prec.update(logits, y)
        self.train_recall.update(logits, y)
        self.train_prec0.update(-logits, 1-y)
        self.train_recall0.update(-logits, 1-y)
        self.train_cm.update(logits, y)

        self.log("train_loss", loss, prog_bar=True)
        
        return loss
    
    def on_train_epoch_end(self):
        self.log("train_acc", self.train_acc.compute(), sync_dist=True)
        self.log("train_recall", self.train_recall.compute(), sync_dist=True)
        self.log("train_recall0", self.train_recall0.compute(), sync_dist=True)
        self.log("train_prec", self.train_prec.compute(), sync_dist=True)
        self.log("train_prec0", self.train_prec0.compute(), sync_dist=True)
        # self.log("train_cm", self.train_cm.compute(), sync_dist=True)

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        y = y.view_as(logits)
        loss = nn.functional.binary_cross_entropy_with_logits(logits, y)
        
        self.val_acc.update(logits, y)
        self.val_prec.update(logits, y)
        self.val_recall.update(logits, y)
        self.val_prec0.update(-logits, 1-y)
        self.val_recall0.update(-logits, 1-y)
        self.val_cm.update(logits, y)

        self.log("val_loss", loss, prog_bar=True, sync_dist=True)
        self.log("val_acc", self.val_acc.compute(), prog_bar=True, sync_dist=True)
        self.log("val_recall", self.val_recall.compute(), prog_bar=True, sync_dist=True)
        self.log("val_recall0", self.val_recall0.compute(), prog_bar=True, sync_dist=True)
        self.log("val_prec", self.val_prec.compute(), sync_dist=True)
        self.log("val_prec0", self.val_prec0.compute(), sync_dist=True)
        # self.log("val_cm", self.val_cm.compute(), sync_dist=True)

        return loss
    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=0.001)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)

        out_dict = {'optimizer': optimizer, 'lr_scheduler': scheduler}

        return out_dict
    

class ETIDataModule(L.LightningDataModule):
    def __init__(self, fn, batch_size):
        super().__init__()
        self.fn = fn
        self.batch_size = batch_size

    def setup(self, stage=None):
        with open(self.fn, 'rb') as f:
            event_hits, event_types = pickle.load(f)
    
        XBins, YBins, ZBins = 110, 110, 48
        XMin, XMax = -55, 55
        YMin, YMax = -55, 55
        ZMin, ZMax = 0, 48

        dims = [XBins, YBins, ZBins]
        xrange = [XMin, XMax]
        yrange = [YMin, YMax]
        zrange = [ZMin, ZMax]

        ranges = [xrange, yrange, zrange]

        dataset_all = VoxelDataset(event_hits, event_types, dims, ranges)

        split = 0.95
        ntrain = int(len(dataset_all)*split)
        nval = len(dataset_all) - ntrain

        self.train, self.val = random_split(
            dataset_all, [ntrain, nval],
            generator=torch.Generator().manual_seed(42)
        )

    def train_dataloader(self):
        return DataLoader(self.train, batch_size=self.batch_size, num_workers=16, persistent_workers=True)
    
    def val_dataloader(self):
        return DataLoader(self.val, batch_size=self.batch_size, num_workers=16, persistent_workers=True)
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Binary Classification of Events')

    parser.add_argument('-fn', dest='fn', action='store', help='Dataset filename')
    # parser.add_argument('-label', dest='label', action='store', default="",
                        # help='Label to add to output data')
    parser.add_argument('-dir', dest='dir', action='store', default="./",
                        help='Directory for output data')
    # parser.add_argument('-model', dest='model', action='store', default='ComPairNet',
                        # help='Model to use')
    parser.add_argument("-batch", dest='batch', action='store', type=int, default=128,
                        help="Batch size")
    
    args = parser.parse_args()
    
    datamodule = ETIDataModule(args.fn, args.batch)
    classifier = EventTypeIdentification()

    if torch.cuda.is_available():
        trainer = L.Trainer(max_epochs=20, accelerator="gpu", devices=2, strategy='ddp', default_root_dir=args.dir)
    else:
        trainer = L.Trainer(max_epochs=10, default_root_dir=args.dir)

    trainer.fit(model=classifier, datamodule=datamodule)