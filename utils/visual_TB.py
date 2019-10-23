import torch
import torchvision
from torch.utils.tensorboard import SummaryWriter


class Visualizer():

    def __init__(self, log_dir):

        self.summary_writer = SummaryWriter(log_dir=log_dir)

    def write_lr(self, optim, globaliter):
        for i, param_group in enumerate(optim.param_groups):
            self.summary_writer.add_scalar('learning_rate/lr_' + str(i), param_group['lr'], globaliter)
        self.summary_writer.flush()

    def write_loss_train(self, value, globaliter):
        self.summary_writer.add_scalar('Loss/train', value, globaliter)
        self.summary_writer.flush()

    def write_loss_validation(self, value, globaliter, if_testtimes=False):
        if if_testtimes:
            postfix = '_testtimes'
        else:
            postfix = ''

        self.summary_writer.add_scalar('Loss/validation' + postfix, value, globaliter)
        self.summary_writer.flush()

    def write_image(self, images, epoch, if_predict=False, if_testtimes=False):

        if if_testtimes:
            postfix = '_testtimes'
        else:
            postfix = ''
        if len(images.shape) == 4:
            _, _, row, col = images.shape
            vol_batch = torch.zeros((3, 1, row, col))
            speed_batch = torch.zeros((3, 1, row, col))
            head_batch = torch.zeros((3, 1, row, col))

            # volume
            vol_batch[0] = images[0, 0, :, :]
            vol_batch[1] = images[0, 3, :, :]
            vol_batch[2] = images[0, 6, :, :]

            # speed
            speed_batch[0] = images[0, 1, :, :]
            speed_batch[1] = images[0, 4, :, :]
            speed_batch[2] = images[0, 7, :, :]

            # heading
            head_batch[0] = images[0, 2, :, :]
            head_batch[1] = images[0, 5, :, :]
            head_batch[2] = images[0, 8, :, :]
        else:
            _, _, _, row, col = images.shape
            vol_batch = torch.zeros((3, 1, row, col))
            speed_batch = torch.zeros((3, 1, row, col))
            head_batch = torch.zeros((3, 1, row, col))

            # volume
            vol_batch[0] = images[0, 0, 0, :, :]
            vol_batch[1] = images[0, 1, 0, :, :]
            vol_batch[2] = images[0, 2, 0, :, :]
            # speed
            speed_batch[0] = images[0, 0, 1, :, :]
            speed_batch[1] = images[0, 1, 1, :, :]
            speed_batch[2] = images[0, 2, 1, :, :]
            # heading
            head_batch[0] = images[0, 0, 2, :, :]
            head_batch[1] = images[0, 1, 2, :, :]
            head_batch[2] = images[0, 2, 2, :, :]

        if if_predict:
            vol_batch = torchvision.utils.make_grid(vol_batch, normalize=True, range=(0, 1))
            self.summary_writer.add_image('prediction' + postfix + '/volume', vol_batch, epoch)

            speed_batch = torchvision.utils.make_grid(speed_batch, normalize=True, range=(0, 1))
            self.summary_writer.add_image('prediction' + postfix + '/speed', speed_batch, epoch)

            head_batch = torchvision.utils.make_grid(head_batch, normalize=True, range=(0, 1))
            self.summary_writer.add_image('prediction' + postfix + '/heading', head_batch, epoch)

        else:
            vol_batch = torchvision.utils.make_grid(vol_batch, normalize=True, range=(0, 1))
            self.summary_writer.add_image('ground_truth' + postfix + '/volume', vol_batch, epoch)

            speed_batch = torchvision.utils.make_grid(speed_batch, normalize=True, range=(0, 1))
            self.summary_writer.add_image('ground_truth' + postfix + '/speed', speed_batch, epoch)

            head_batch = torchvision.utils.make_grid(head_batch, normalize=True, range=(0, 1))
            self.summary_writer.add_image('ground_truth' + postfix + '/heading', head_batch, epoch)

        self.summary_writer.flush()

    def close(self):
        self.summary_writer.close()


if __name__ == "__main__":

    import numpy as np

    img_batch = np.zeros((16, 3, 100, 100))
    for i in range(16):
        img_batch[i, 0] = np.arange(0, 10000).reshape(100, 100) / 10000 / 16
        print(img_batch[i, 0].shape)

        img_batch[i, 1] = (1 - np.arange(0, 10000).reshape(100, 100) / 10000) / 16

    print(img_batch.shape)
