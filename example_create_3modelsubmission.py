import json
import os
import time

import torch

from models.unet import UNet
from utils.create_submission import submission_writer
from utils.videoloader import trafic4cast_dataset

padd5 = torch.nn.ZeroPad2d((6, 6, 1, 0))


def unpadd5(x):
    return x[:, :, 1:, 6:-6]


def predict_submit(loader, model, device,
                   submission_root=os.path.join('submission', 'unet5_clamp_mask'),
                   print_interval=10, reduce=False, writer_kwds={}):
    sw = submission_writer(loader.dataset, batch_size=loader.batch_size,
                           submission_root=submission_root, **writer_kwds)
    model.eval()
    with torch.no_grad():

        n_batches = len(loader)

        start_time = time.time()

        for idx, (inputs, target, feat_dict) in enumerate(loader, 0):
            inputs = padd5(inputs.to(device))

            prediction = model(inputs)

            prediction = unpadd5(prediction)
            prediction = torch.clamp(prediction, 0, 255)

            sw.write_submission_data(prediction.cpu(), idx)

            if (idx + 1) % (print_interval + 1) == 0:
                print("{:d}% \t took: {:.2f}s".format(
                    int(100 * (idx + 1) / n_batches), time.time() - start_time))


if __name__ == "__main__":

    # list of three models for 3 cities
    model_path_list = [os.path.join('runs', 'UNet_Berlin'),
                       os.path.join('runs', 'Unet_Istanbul'),
                       os.path.join('runs', 'Unet_Moscow')]

    submission_root = os.path.join("submission", "Unet")
    print_interval = 1

    writer_kwds = {'init_dir': True}

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    for model_path in model_path_list:
        config_path = os.path.join(model_path, 'config.json')
        with open(config_path, 'r') as f:
            config = json.load(f)

        kwds_dataset = config['dataset']
        kwds_dataset['split_type'] = 'test'

        kwds_loader = config['dataloader']
        kwds_loader['batch_size'] = 5
        kwds_loader['shuffle'] = False

        reduce = config['dataset']['reduce']

        writer_kwds['cities'] = config['dataset']['cities']

        model = UNet(**config['model']).to(device)
        model.load_state_dict(torch.load(os.path.join(model_path, 'checkpoint.pt'), map_location=device))

        dataset = trafic4cast_dataset(**config['dataset'])
        loader = torch.utils.data.DataLoader(dataset, **kwds_loader)

        predict_submit(loader, model, device, submission_root=submission_root, print_interval=print_interval,
                       reduce=reduce, writer_kwds=writer_kwds)

    print('done')
