import os
import json
import pickle
import argparse

import numpy as np
from tqdm import tqdm
from torch.utils.data import Dataset
import ipdb
from preprocess import pad_2D

num_joint = 14
max_frame = 60
# max_frame = 300
# num_person_out = 2
# num_person_in = 5
num_person_out = 1
num_person_in = 2


class Feeder_kinetics(Dataset):
    """ Feeder for skeleton-based action recognition in kinetics-skeleton dataset
    # Joint index:
    # {0,  "Nose"}
    # {1,  "Neck"},
    # {2,  "RShoulder"},
    # {3,  "RElbow"},
    # {4,  "RWrist"},
    # {5,  "LShoulder"},
    # {6,  "LElbow"},
    # {7,  "LWrist"},
    # {8,  "RHip"},
    # {9,  "RKnee"},
    # {10, "RAnkle"},
    # {11, "LHip"},
    # {12, "LKnee"},
    # {13, "LAnkle"},
    # {14, "REye"},
    # {15, "LEye"},
    # {16, "REar"},
    # {17, "LEar"},
    Arguments:
        data_path: the path to '.npy' data, the shape of data should be (N, C, T, V, M)
        label_path: the path to label
        window_size: The length of the output sequence
        num_person_in: The number of people the feeder can observe in the input sequence
        num_person_out: The number of people the feeder in the output sequence
        debug: If true, only use the first 100 samples
    """

    def __init__(self,
                 data_path,
                 label_path,
                 ignore_empty_sample=True,
                 window_size=-1,
                 num_person_in=5,
                 num_person_out=2):
        self.data_path = data_path
        self.label_path = label_path
        self.window_size = window_size
        self.num_person_in = num_person_in
        self.num_person_out = num_person_out
        self.ignore_empty_sample = ignore_empty_sample

        self.load_data()

    def load_data(self):
        # ipdb.set_trace()
        # load file list
        self.sample_name = os.listdir(self.data_path)

        # load label
        label_path = self.label_path
        with open(label_path) as f:
            label_info = json.load(f)

        sample_id = [name.split('.')[0] for name in self.sample_name]
        # ipdb.set_trace()
        self.label = np.array([label_info[id]['label_index'] for id in sample_id])
        has_skeleton = np.array([label_info[id]['has_skeleton'] for id in sample_id])

        # ignore the samples which does not has skeleton sequence
        # if self.ignore_empty_sample:
        #     self.sample_name = [s for h, s in zip(has_skeleton, self.sample_name) if h]
        #     self.label = self.label[has_skeleton]

        # output data shape (N, C, T, V, M)
        self.N = len(self.sample_name)  # sample
        self.C = 3  # channel
        self.T = max_frame  # frame
        self.V = num_joint  # joint
        self.M = self.num_person_out  # person

    def __len__(self):
        return len(self.sample_name)

    def __iter__(self):
        return self

    def __getitem__(self, index):

        # output shape (C, T, V, M)
        # get data
        # ipdb.set_trace()
        sample_name = self.sample_name[index]
        sample_path = os.path.join(self.data_path, sample_name)
        with open(sample_path, 'r') as f:
            video_info = json.load(f)

        # fill data_numpy
        # if total frame < T, fill with 0
        data_numpy = np.zeros((self.C, self.T, self.V, self.num_person_in))
        for frame_info in video_info['data']:
            frame_index = frame_info['frame_index']
            if frame_index > max_frame - 1:
                break
            for m, skeleton_info in enumerate(frame_info["skeleton"]):
                # openpose estimate m person skeleton
                if m >= self.num_person_in:
                    break
                pose = skeleton_info['pose']
                score = skeleton_info['score']
                data_numpy[0, frame_index, :, m] = pose[0::2]
                data_numpy[1, frame_index, :, m] = pose[1::2]
                data_numpy[2, frame_index, :, m] = score

        # centralization
        # ipdb.set_trace()
        data_numpy[0:2] = data_numpy[0:2] - 0.5
        # data_numpy[1:2] = -data_numpy[1:2] don't need
        data_numpy[0][data_numpy[2] == 0] = 0
        data_numpy[1][data_numpy[2] == 0] = 0

        # ipdb.set_trace()

        # get & check label index
        label = video_info['label_index']
        assert (self.label[index] == label)

        # sort by score
        sort_index = (-data_numpy[2, :, :, :].sum(axis=1)).argsort(axis=1)
        for t, s in enumerate(sort_index):
            data_numpy[:, t, :, :] = data_numpy[:, t, :, s].transpose((1, 2,
                                                                       0))

        # if openpose estimate multi-person (m), only use 2 skeleton
        data_numpy = data_numpy[:, :, :, 0:self.num_person_out]

        return data_numpy, label


def gendata(data_path, label_path,
            data_out_path, label_out_path,
            num_person_in=num_person_in,  # observe the first 5 persons
            num_person_out=num_person_out,  # then choose 2 persons with the highest score
            max_frame=max_frame):
    feeder = Feeder_kinetics(
        data_path=data_path,
        label_path=label_path,
        num_person_in=num_person_in,
        num_person_out=num_person_out,
        window_size=max_frame)

    # only 10 classs and negative class
    # dicts = {42:0, 26:1, 13:2, 8:3, 21:4, 23:5, 14:6, 25:7, 7:8, 41:9}
    dicts = {42:0, 26:1, 8:2, 21:3, 23:4, 25:5, 7:6, 41:7} # remove long time

    # random 10 classa and negative class
    # dicts = {56:0, 33:1, 5:2, 19:3, 56:4, 32:5, 22:6, 24:7, 35:8, 58:9}
    # dicts = {38: 0, 42: 1, 23: 2, 19: 3, 43: 4, 44: 5, 8: 6, 55: 7, 56: 8, 10: 9}
    # random 10 classa and negative class one person
    # dicts = {38: 0, 42: 1, 23: 2, 19: 3, 43: 4, 44: 5, 8: 6, 9: 7, 37: 8, 10: 9}
    sample_name = feeder.sample_name
    sample_label = []

    fp = np.zeros((len(sample_name), 3, max_frame, num_joint, num_person_out), dtype=np.float32)

    for i, s in enumerate(tqdm(sample_name)):
        data, label = feeder[i]
        fp[i, :, 0:data.shape[1], :, :] = data
        if label in dicts:
            sample_label.append(dicts[label])
        else:
            sample_label.append(8)
        # sample_label.append(label)

    with open(label_out_path, 'wb') as f:
        pickle.dump((sample_name, list(sample_label)), f)

    fp = pad_2D(fp)
    np.save(data_out_path, fp)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Kinetics-skeleton Data Converter.')
    parser.add_argument(
        '--data_path', default='/6T-4/fengyu/output')
    parser.add_argument(
        # '--out_folder', default='/6T-4/fengyu/output/ntu2D')
        '--out_folder', default = '/6T-4/fengyu/output/ntu2D_60') # for random
    arg = parser.parse_args()

    # part = ['train']
    part = ['val', 'train']
    for p in part:
        print('ntu2D ', p)
        if not os.path.exists(arg.out_folder):
            os.makedirs(arg.out_folder)
        data_path = '{}/ntu_{}'.format(arg.data_path, p)
        label_path = '{}/label/{}_label.json'.format(arg.data_path, p)
        data_out_path = '{}/{}_data_joint.npy'.format(arg.out_folder, p)
        label_out_path = '{}/{}_label.pkl'.format(arg.out_folder, p)

        gendata(data_path, label_path, data_out_path, label_out_path)
