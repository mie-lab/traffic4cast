
config = dict()
##################################################################
# data set configuration
# source_root = raw data
# target_root = folder to store preprocessed data (preprocessing only done once)

config['dataset'] = {}
config['dataset']['source_root'] = 'data_raw'
config['dataset']['target_root'] = 'data'
config['dataset']['reduce'] = True
config['dataset']['return_features'] = False
config['dataset']['cities'] = ['Berlin']  # has to be iterable

# The compression argument is only important for data preprocessing (the very first run)
# compression can be 'lzf' or None. None increases speed but needs ~200GB of storage
config['dataset']['compression'] = None
##################################################################

config['device_num'] = 0
config['debug'] = False

# model statistics 
config['model'] = {}
config['model']['in_channels'] = 36
config['model']['n_classes'] = 9
config['model']['depth'] = 5
config['model']['wf'] = 6
config['model']['padding'] = True
config['model']['up_mode'] = 'upconv'  # up_mode (str): 'upconv' or 'upsample'.
config['model']['batch_norm'] = True

config['cont_model_path'] = None  # Use this to continue training a previously started model.

# data loader configuration
config['dataloader'] = {}
config['dataloader']['drop_last'] = True
config['dataloader']['shuffle'] = True
config['dataloader']['num_workers'] = 4
config['dataloader']['batch_size'] = 3

# optimizer
config['optimizer'] = {}
config['optimizer']['lr'] = 1e-2
config['optimizer']['weight_decay'] = 0
config['optimizer']['momentum'] = 0.9
config['optimizer']['nesterov'] = True

# lr schedule
config['lr_step_size'] = 5
config['lr_gamma'] = 0.1

# early stopping
config['patience'] = 10

config['num_epochs'] = 50
config['print_every_step'] = 50
