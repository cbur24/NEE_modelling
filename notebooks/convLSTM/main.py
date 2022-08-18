# import libraries
import os
import matplotlib.pyplot as plt
import torch
import torchvision
from torch.nn import functional as F
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from pytorch_lightning import Trainer
from multiprocessing import Process

import sys
sys.path.append('/g/data/os22/chad_tmp/NEE_modelling/notebooks/convLSTM/')
from start_tensorboard import run_tensorboard
from seq2seq_ConvLSTM import EncoderDecoderConvLSTM
import xarray as xr

import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--lr', default=1e-4, type=float, help='learning rate')
parser.add_argument('--beta_1', type=float, default=0.9, help='decay rate 1')
parser.add_argument('--beta_2', type=float, default=0.98, help='decay rate 2')
parser.add_argument('--batch_size', default=12, type=int, help='batch size')
parser.add_argument('--epochs', type=int, default=300, help='number of epochs to train for')
parser.add_argument('--use_amp', default=False, type=bool, help='mixed-precision training')
parser.add_argument('--n_gpus', type=int, default=1, help='number of GPUs')
parser.add_argument('--n_hidden_dim', type=int, default=64, help='number of hidden dim for ConvLSTM layers')

opt = parser.parse_args()


class XarrayDataset(Dataset):
    '''A simple torch Dataset wrapping xr.DataArray'''
    def __init__(self, ar, batch_dim):
        self.ar = ar
        self.batch_dim = batch_dim

    def __len__(self):
        return len(self.ar[self.batch_dim])

    def __getitem__(self, idx):
        return self.ar[{self.batch_dim: idx}].values

class XarrayDataLoader(DataLoader):
    '''A simple torch DataLoader wrapping xr.DataArray'''
    def __init__(self, ar, batch_dim, **kwargs):
        ar = XarrayDataset(ar, batch_dim)
        super().__init__(ar, **kwargs)



##########################
######### MODEL ##########
##########################

class MovingMNISTLightning(pl.LightningModule):

    def __init__(self, hparams=None, model=None):
        super(MovingMNISTLightning, self).__init__()

        # default config
        self.path = '/g/data/os22/chad_tmp/NEE_modelling/notebooks/convLSTM/mnist_test_seq.npy'
        self.model = model

        # logging config
        self.log_images = True

        # Training config
        self.train_test_split = 0.8
        self.criterion = torch.nn.MSELoss()
        self.batch_size = opt.batch_size
        self.n_steps_past = 10
        self.n_steps_ahead = 10  # 4

    def create_video(self, x, y_hat, y):
        # predictions with input for illustration purposes
        preds = torch.cat([x.cpu(), y_hat.unsqueeze(2).cpu()], dim=1)[0]

        # entire input and ground truth
        y_plot = torch.cat([x.cpu(), y.unsqueeze(2).cpu()], dim=1)[0]

        # error (l2 norm) plot between pred and ground truth
        difference = (torch.pow(y_hat[0] - y[0], 2)).detach().cpu()
        zeros = torch.zeros(difference.shape)
        difference_plot = torch.cat([zeros.cpu().unsqueeze(0), difference.unsqueeze(0).cpu()], dim=1)[
            0].unsqueeze(1)

        # concat all images
        final_image = torch.cat([preds, y_plot, difference_plot], dim=0)

        # make them into a single grid image file
        grid = torchvision.utils.make_grid(final_image, nrow=self.n_steps_past + self.n_steps_ahead)

        return grid

    def forward(self, x):
        x = x.to(device='cuda')

        output = self.model(x, future_seq=self.n_steps_ahead)

        return output

    def training_step(self, batch, batch_idx):
        x, y = batch[:, 0:self.n_steps_past, :, :, :], batch[:, self.n_steps_past:, :, :, :]
        x = x.permute(0, 1, 4, 2, 3)
        y = y.squeeze()

        y_hat = self.forward(x).squeeze()  # is squeeze neccessary?

        loss = self.criterion(y_hat, y)

        # save learning_rate
        lr_saved = self.trainer.optimizers[0].param_groups[-1]['lr']
        lr_saved = torch.scalar_tensor(lr_saved).cuda()

        # save predicted images every 250 global_step
        if self.log_images:
            if self.global_step % 250 == 0:
                final_image = self.create_video(x, y_hat, y)

                self.logger.experiment.add_image(
                    'epoch_' + str(self.current_epoch) + '_step' + str(self.global_step) + '_generated_images',
                    final_image, 0)
                plt.close()

        tensorboard_logs = {'train_mse_loss': loss,
                            'learning_rate': lr_saved}

        return {'loss': loss, 'log': tensorboard_logs}


    def test_step(self, batch, batch_idx):
        # OPTIONAL
        x, y = batch
        y_hat = self.forward(x)
        return {'test_loss': self.criterion(y_hat, y)}


    def test_end(self, outputs):
        # OPTIONAL
        avg_loss = torch.stack([x['test_loss'] for x in outputs]).mean()
        tensorboard_logs = {'test_loss': avg_loss}
        return {'avg_test_loss': avg_loss, 'log': tensorboard_logs}


    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=opt.lr, betas=(opt.beta_1, opt.beta_2))

    @pl.data_loader
    def train_dataloader(self):
        
        # Load Data as Numpy Array
        dataset = np.load(self.path).transpose(1, 0, 2, 3)
        
        # Split into train and validation sets using indexing to optimize memory.
        indexes = np.arange(dataset.shape[0])
        np.random.shuffle(indexes)
        train_index = indexes[: int(self.train_test_split * dataset.shape[0])]
        train_dataset = dataset[train_index]

        def collate(batch):

            # Add channel dim, scale pixels between 0 and 1, send to GPU
            batch = torch.tensor(batch).unsqueeze(1)     
            batch = batch / 255.0                        
            batch = batch.to(device)                     

            # Randomly pick 10 frames as input, 11th frame is target
            rand = np.random.randint(10,20)                     
            return batch[:,:,rand-10:rand], batch[:,:,rand]     

        # Training Data Loader
        train_loader = DataLoader(train_dataset,
                                  shuffle=True, 
                                  batch_size=self.batch_size,
                                  collate_fn=collate
                                 )

        return train_loader

    @pl.data_loader
    def test_dataloader(self):
        
        # Load Data as Numpy Array
        dataset = np.load(self.path).transpose(1, 0, 2, 3)
        
        # Split into train and validation sets using indexing to optimize memory.
        indexes = np.arange(dataset.shape[0])
        np.random.shuffle(indexes)
        val_index = indexes[int(train_test_split * dataset.shape[0]) :]
        val_dataset = dataset[val_index]

        def collate(batch):

            # Add channel dim, scale pixels between 0 and 1, send to GPU
            batch = torch.tensor(batch).unsqueeze(1)     
            batch = batch / 255.0                        
            batch = batch.to(device)                     

            # Randomly pick 10 frames as input, 11th frame is target
            rand = np.random.randint(10,20)                     
            return batch[:,:,rand-10:rand], batch[:,:,rand]     

        # Training Data Loader
        test_loader = DataLoader(train_data,
                                  shuffle=True, 
                                  batch_size=self.batch_size,
                                  collate_fn=collate)



def run_trainer():
    conv_lstm_model = EncoderDecoderConvLSTM(nf=opt.n_hidden_dim, in_chan=1)

    model = MovingMNISTLightning(model=conv_lstm_model)

    trainer = Trainer(max_epochs=opt.epochs,
                      gpus=opt.n_gpus,
                      distributed_backend='dp',
                      early_stop_callback=False,
                      use_amp=opt.use_amp
                      )

    trainer.fit(model)


if __name__ == '__main__':
    p1 = Process(target=run_trainer)                    # start trainer
    p1.start()
    p2 = Process(target=run_tensorboard(new_run=True))  # start tensorboard
    p2.start()
    p1.join()
    p2.join()



