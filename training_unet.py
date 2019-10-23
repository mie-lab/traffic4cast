import json
import os
import time
import warnings
from datetime import datetime

import torch
from torch.optim.lr_scheduler import StepLR

from config_unet import config
from models.unet import UNet
from utils.earlystopping import EarlyStopping
from utils.videoloader import trafic4cast_dataset

with warnings.catch_warnings():  # pytorch tensorboard throws future warnings until the next update
    warnings.filterwarnings("ignore", category=FutureWarning)
    from utils.visual_TB import Visualizer


def trainNet(model, train_loader, val_loader, val_loader_ttimes, device):
    # Print all of the hyper parameters of the training iteration:
    print("===== HYPERPARAMETERS =====")
    print("batch_size=", config['dataloader']['batch_size'])
    print("epochs=", config['num_epochs'])
    print("learning_rate=", config['optimizer']['lr'])
    print("network_depth=", config['model']['depth'])
    print("=" * 30)

    # define the optimizer & learning rate 
    optim = torch.optim.SGD(model.parameters(), **config['optimizer'])

    scheduler = StepLR(optim,
                       step_size=config['lr_step_size'],
                       gamma=config['lr_gamma'])

    if config['cont_model_path'] is not None:
        log_dir = config['cont_model_path']
    else:
        log_dir = 'runs/Unet-' + datetime.now().strftime("%Y-%m-%d-%H-%M-%S-") + \
                  '-'.join(config['dataset']['cities'])
    writer = Visualizer(log_dir)

    # dump config file
    with open(os.path.join(log_dir, 'config.json'), 'w') as fp:
        json.dump(config, fp)

    # Time for printing
    training_start_time = time.time()
    globaliter = 0

    # initialize the early_stopping object
    early_stopping = EarlyStopping(log_dir, patience=config['patience'], verbose=True)

    # Loop for n_epochs
    for epoch_idx, epoch in enumerate(range(config['num_epochs'])):
        writer.write_lr(optim, epoch)

        # train for one epoch
        globaliter = train(model, train_loader, optim, device, writer, epoch, globaliter)

        # At the end of the epoch, do a pass on the validation set
        _ = validate(model, val_loader, device, writer, globaliter)

        # At the end of the epoch, do a pass on the validation set only considering the test times
        val_loss_testtimes = validate(model, val_loader_ttimes, device, writer, globaliter, if_testtimes=True)

        # early_stopping needs the validation loss to check if it has decresed, 
        # and if it has, it will make a checkpoint of the current model
        early_stopping(val_loss_testtimes, model)

        if early_stopping.early_stop:
            print("Early stopping")
            break

        if config['debug'] and epoch_idx >= 0:
            break

        scheduler.step(epoch)

    print("Training finished, took {:.2f}s".format(time.time() - training_start_time))

    # remember to close tensorboard writer
    writer.close()


def train(model, train_loader, optim, device, writer, epoch, globaliter):
    model.train()
    running_loss = 0.0
    n_batches = len(train_loader)

    # define start time
    start_time = time.time()

    for i, data in enumerate(train_loader, 0):
        inputs, Y, feature_dict = data
        inputs = inputs / 255
        globaliter = globaliter + 1

        # padd the input data with 0 to ensure same size after upscaling by the network
        # inputs [495, 436] -> [496, 448] 
        padd = torch.nn.ZeroPad2d((6, 6, 1, 0))

        inputs = padd(inputs)
        inputs = inputs.float().to(device)

        # the Y remains the same dimension
        Y = Y.float().to(device)

        # Set the parameter gradients to zero
        optim.zero_grad()

        prediction = model(inputs)

        # crop the output for comparing with true Y
        loss_size = torch.nn.functional.mse_loss(prediction[:, :, 1:, 6:-6], Y)

        loss_size.backward()
        optim.step()

        # Print statistics
        running_loss += loss_size.item()
        if (i + 1) % config['print_every_step'] == 0:
            print("Epoch {}, {:d}% \t train_loss: {:.3f} took: {:.2f}s".format(
                epoch + 1, int(100 * (i + 1) / n_batches), running_loss / config['print_every_step'],
                time.time() - start_time))

            # write the train loss to tensorboard
            running_loss_norm = running_loss / config['print_every_step']
            writer.write_loss_train(running_loss_norm, globaliter)
            # Reset running loss and time
            running_loss = 0.0
            start_time = time.time()

        if config['debug'] and i >= 0:
            break

    return globaliter


def validate(model, val_loader, device, writer, globaliter, if_testtimes=False):
    total_val_loss = 0
    if if_testtimes:
        prefix = 'testtimes'
    else:
        prefix = ''

    # change to validation mode
    model.eval()
    with torch.no_grad():
        for i, data in enumerate(val_loader, 0):
            val_inputs, val_y, feature_dict = data
            val_inputs = val_inputs / 255

            padd = torch.nn.ZeroPad2d((6, 6, 1, 0))
            val_inputs = padd(val_inputs)

            val_inputs = val_inputs.float().to(device)
            val_y = val_y.float().to(device)

            val_output = model(val_inputs)

            # crop the output for comparing with true Y
            val_loss_size = torch.nn.functional.mse_loss(val_output[:, :, 1:, 6:-6], val_y)
            total_val_loss += val_loss_size.item()

            # each epoch select one prediction set (one batch) to visualize
            if (i + 1) % int(len(val_loader) / 2 + 1) == 0:
                writer.write_image(val_output.cpu(), globaliter, if_predict=True,
                                   if_testtimes=if_testtimes)
                writer.write_image(val_y.cpu(), globaliter, if_predict=False,
                                   if_testtimes=if_testtimes)

            if config['debug'] and i >= 0:
                break

    val_loss = total_val_loss / len(val_loader)
    print("Validation loss {} = {:.2f}".format(prefix, val_loss))

    # write the validation loss to tensorboard
    writer.write_loss_validation(val_loss, globaliter, if_testtimes=if_testtimes)
    return val_loss


if __name__ == "__main__":

    dataset_train = trafic4cast_dataset(split_type='training', **config['dataset'])
    dataset_val = trafic4cast_dataset(split_type='validation', **config['dataset'])
    dataset_val_ttimes = trafic4cast_dataset(split_type='validation', **config['dataset'], filter_test_times=True)

    train_loader = torch.utils.data.DataLoader(dataset_train, **config['dataloader'])
    val_loader = torch.utils.data.DataLoader(dataset_val, **config['dataloader'])
    val_loader_ttimes = torch.utils.data.DataLoader(dataset_val_ttimes, **config['dataloader'])

    device = torch.device(config['device_num'] if torch.cuda.is_available() else 'cpu')

    if config['cont_model_path'] is None:
        # define the network structure -- UNet
        # the output size is not always equal to your input size !!!
        model = UNet(**config['model']).to(device)

    # load alrady trained model
    elif os.path.isdir(config['cont_model_path']):
        cont_model_path = config['cont_model_path']
        current_optimizer = config['optimizer']

        with open(os.path.join(config['cont_model_path'], 'config.json'), 'r') as f:
            model_config = json.load(f)
            # We overwrite this to be able to configure the optimizer on subsequent runs.
            config['model'] = model_config['model']
        model = UNet(**config['model']).to(device)
        model.load_state_dict(torch.load(os.path.join(cont_model_path, 'checkpoint.pt'),
                                         map_location=device))
    else:
        raise Exception(f"Model to continue training not found: {config['cont_model_path']}.")

    # # need to add the mask parameter when training the partial Unet model
    trainNet(model, train_loader, val_loader, val_loader_ttimes, device)
