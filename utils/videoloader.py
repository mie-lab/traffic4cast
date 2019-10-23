import sys, os, time 
from pathlib import Path
import pickle
import datetime as dt
import glob

import h5py

import numpy as np
from matplotlib import pyplot as plt

from multiprocessing import Pool
from functools import partial

import torch
from torchvision import datasets, transforms


def subsample(x, n=0, m=200):
    return x[..., n:m, n:m]

def _get_tstamp_string(tstamp_ix):
    """Calculates the timestamp in hh:mm based on the file index
    
    Args:
        tstamp_ix (int): Index of the single frame  
    
    Returns:
        Str: hh:mm
    """
    total_minutes = tstamp_ix*5
    hours = total_minutes // 60
    minutes = total_minutes % 60
    
    return hours, minutes


class trafic4cast_dataset(torch.utils.data.Dataset):
    """Dataloader for trafic4cast data
    
    Attributes:
        compression (TYPE): Description
        do_precomp_path (TYPE): Description
        num_frames (TYPE): Description
        reduce (TYPE): Description
        source_root (TYPE): Description
        split_type (TYPE): Description
        target_file_paths (TYPE): Description
        target_root (TYPE): Description
        transform (TYPE): Description
        valid_test_clips (TYPE): Description
    """
    
    def __init__(self, source_root, target_root="precomuted_data",
                 split_type='train',
                 cities=['Berlin', 'Istanbul', 'Moscow'],
                 transform=None, reduce=False, compression=None,
                 num_frames=15, do_subsample=None, filter_test_times=False,
                 return_features=False, return_city=False):
        """Dataloader for the trafic4cast competition
        Usage Dataloader:
        The dataloader is situated in "videoloader.py", to use it, you have
        to download the competition data and set two paths. "source_root" 
        and "target_root". 
            source_root: Is the directory with the raw competition data. 
                The expected file structure is shown below.
            target_root: This directory will be used to store the
                preprocessed data (about 200 GB)

        Expected folder structure for raw data:

            -source_root
                - Berlin
                    -Berlin_test
                    -Berlin_training
                    -Berlin_validation
                -Istanbul
                    -Instanbul_test
                    -…
                -Moscow
                    -…
        
        Args:
            source_root (str): Is the directory with the raw competition data.
            target_root (str, optional): This directory will be used to store the
                preprocessed data
            split_type (str, optional): Can be ['training', 'validation', 'test']
            cities (list, optional): This can be used to limit the data loader to a
                subset of cities. Has to be a list! Default is ['Berlin', 'Moscow', 'Istanbul']
            transform (None, optional): Transform applied to x before returning it.
            reduce (bool, optional): This option collapses the time dimension into the
                (color) channel dimension.
            compression (str, optional): The h5py compression method to store the 
                preprocessed data. 'compression=None' is the fastest. 
            num_frames (int, optional): 
            do_subsample (tuple, optional): Tuple of two integers. Returns only a part of the image. Slices the 
                image in the 'pixel' dimensions with x = x[n:m, n:m]. with m>n
            filter_test_times (bool, optional): Filters output data, such that only valid (city-dependend) test-times are returned. 
        """
        self.reduce = reduce
        self.source_root = source_root
        self.target_root = target_root
        self.transform = transform
        self.split_type = split_type
        self.compression = compression
        self.cities = cities
        self.num_frames = num_frames
        self.subsample = False
        self.filter_test_times = filter_test_times
        self.return_features = return_features
        self.return_city = return_city

        if self.filter_test_times:
            tt_dict2 = {}
            tt_dict = pickle.load(open(os.path.join('.', 'utils', 'test_timestamps.dict'), "rb"))
            for city, values in tt_dict.items():
                values.sort()
                tt_dict2[city] = values
                
            self.valid_test_times = tt_dict2
        
        if do_subsample is not None:
            self.subsample = True
            self.n = do_subsample[0]
            self.m = do_subsample[1]

        source_file_paths = []
        for city in cities:
            source_file_paths = source_file_paths + glob.glob(
                os.path.join(self.source_root, city, '*_' + self.split_type,
                             '*.h5'))

        do_precomp_path = []
        missing_target_files = []
        for raw_file_path in source_file_paths:
            target_file = raw_file_path.replace(
                self.source_root, self.target_root)
            if not os.path.exists(target_file):
                do_precomp_path.append(raw_file_path)
                missing_target_files.append(target_file)

        self.do_precomp_path = do_precomp_path

        target_dirs = list(set([str(Path(x).parent)
                                for x in missing_target_files]))
        for target_dir in target_dirs:
            if not os.path.exists(target_dir):
                os.makedirs(target_dir)

        with Pool() as pool:
            pool.map(self.precompute_clip, self.do_precomp_path)
            pool.close()
            pool.join()

        target_file_paths = []
        for city in cities:
            target_file_paths = target_file_paths + glob.glob(
                os.path.join(self.target_root, city, '*_' + self.split_type,
                             '*.h5'))

        self.target_file_paths = target_file_paths

        if self.split_type == 'test':
            precomp_readt_test = partial(self.precompute_clip, mode='reading_test')
            with Pool() as pool:

                valid_test_clips = pool.map(precomp_readt_test,
                                            self.target_file_paths)
                pool.close()
                pool.join()

            valid_test_clips = [valid_tuple for sublist in valid_test_clips
                                for valid_tuple in sublist]
            valid_test_clips.sort()
            self.valid_test_clips = valid_test_clips

    def precompute_clip(self, source_path, mode='writing'):
        """Summary
        
        Args:
            source_path (TYPE): Description
            mode (str, optional): Description
        
        Returns:
            TYPE: Description
        """
        target_path = source_path.replace(self.source_root, self.target_root)

        f_source = h5py.File(source_path, 'r')
        data1 = f_source['array']
        data1 = data1[:]

        if mode == 'writing':

            data1 = np.moveaxis(data1, 3, 1)
            f_target = h5py.File(target_path, 'w')
            dset = f_target.create_dataset('array', (288, 3, 495, 436),
                                           chunks=(1, 3, 495, 436),
                                           dtype='uint8', data=data1,
                                           compression=self.compression)

            f_target.close()

        if mode == 'reading_test':
            valid_test_clips = list = []

            for tstamp_ix in range(288-15):
                clip = data1[tstamp_ix:tstamp_ix+self.num_frames, :, :, :]
                sum_first_train_frame = np.sum(clip[0, :, :, :])
                sum_last_train_frame = np.sum(clip[11, :, :, :])

                if (sum_first_train_frame != 0) and (sum_last_train_frame != 0):
                    valid_test_clips.append((source_path, tstamp_ix))

        f_source.close()

        if mode == 'reading_test':
            return valid_test_clips

    def __len__(self):
        if self.split_type == 'test':
            pass
            return len(self.valid_test_clips)

        elif self.filter_test_times:
            return len(self.target_file_paths) * 5

        else:
            return len(self.target_file_paths) * 272
        

    def __getitem__(self, idx):
        """Summary
        
        Args:
            idx (TYPE): Description
        
        Returns:
            TYPE: Description
        """
        return_dict = {}

        if torch.is_tensor(idx):
            idx = idx.tolist()

        if self.split_type == 'test':
            target_file_path, tstamp_ix = self.valid_test_clips[idx]

        elif self.filter_test_times:
            file_ix = idx // 5
            valid_tstamp_ix = idx % 5
            
            target_file_path = self.target_file_paths[file_ix]
            city_name_path = Path(target_file_path.replace(self.target_root,''))
            city_name = city_name_path.parts[1]
            
            tstamp_ix =  self.valid_test_times[city_name][valid_tstamp_ix]

        else:
            file_ix = idx // 272
            tstamp_ix = idx % 272
            target_file_path = self.target_file_paths[file_ix]
        
        if self.return_features:
            # create feature vector
            date_string = Path(target_file_path).name.split('_')[0]
            date_datetime = dt.datetime.strptime(date_string, '%Y%m%d')
            
            hour, minute = _get_tstamp_string(tstamp_ix)
            
    #        feature_vector = []
            sin_hours = np.sin(2*np.pi/24 * hour)
            cos_hours = np.cos(2*np.pi/24 * hour)
            sin_mins = np.sin(2*np.pi/60 * minute)
            cos_mins = np.cos(2*np.pi/60 * minute)
            
            
            sin_month = np.sin(2*np.pi/12 * date_datetime.month)
            cos_month = np.cos(2*np.pi/12 * date_datetime.month)
            weekday_ix =  date_datetime.weekday() / 6
            week_number = date_datetime.isocalendar()[1] / 52
            
            feature_vector = np.asarray([sin_hours, cos_hours, sin_mins,
                                         cos_mins, sin_month, cos_month, 
                                         weekday_ix, week_number]).ravel()
            feature_vector = torch.from_numpy(feature_vector)
            feature_vector = feature_vector.to(dtype=torch.float)

            return_dict['feature_vector'] = feature_vector

        if self.return_city:
            city_name_path = Path(target_file_path.replace(self.target_root,''))
            city_name = city_name_path.parts[1]
            return_dict['city_names'] = city_name


        
       

        # we want to predict the image at idx+1 based on the image with idx
        f = h5py.File(target_file_path, 'r')
        sample = f.get('array')
        
        x = sample[tstamp_ix:tstamp_ix+12, :, :, :]
        y = sample[tstamp_ix+12:tstamp_ix+15, :, :, :]

        if self.reduce:
            # stack all time dimensions into the channels.
            # all channels of the same timestamp are left togehter
            x = np.moveaxis(x, (0, 1), (2, 3))
            x = np.reshape(x, (495, 436, 36))
            x = torch.from_numpy(x)
            x = x.permute(2, 0, 1)  # Dimensions: time/channels, h, w


            y = np.moveaxis(y, (0, 1), (2, 3))
            y = np.reshape(y, (495, 436, 9))

            y = torch.from_numpy(y)
            y = y.permute(2, 0, 1)
            
            y = y.to(dtype=torch.float)  # is ByteTensor?
            x = x.to(dtype=torch.float)  # is ByteTensor?

        else:

            x = torch.from_numpy(x)
            y = torch.from_numpy(y)

            y = y.to(dtype=torch.float)  # is ByteTensor?
            x = x.to(dtype=torch.float)  # is ByteTensor?

        f.close()
        
        if self.subsample:
            x = subsample(x,self.n,self.m)
            y = subsample(y,self.n,self.m)
        
      
        
        
        if self.transform is not None:
            x =  self.transform(x)

        return x, y, return_dict