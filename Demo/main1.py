#!/usr/bin/env python
from __future__ import print_function
import os
import time
import yaml
import pprint
import random
import pickle
import shutil
import inspect
import argparse
from collections import OrderedDict, defaultdict
import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from tensorboardX import SummaryWriter
from torch.optim.lr_scheduler import MultiStepLR
from utils import count_params, import_class
import ipdb
import tools.utils as utils
import tools.demo_realtime as realtime
import skvideo.io
import sys
import cv2
import tools.get_raw_skes_data as ntu_skes
sys.path.append('./simpleHRNet')
from simpleHRNet.processbyhrnet0 import convert_to_universal
from simpleHRNet.processbyhrnet0 import HRNetProcessor
from tools.preprocess import pre_normalization, pad_2D
import copy
import onnxruntime as ort
import numpy as np

def init_seed(seed):
    torch.cuda.manual_seed_all(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


def get_parser():
    # parameter priority: command line > config file > default
    parser = argparse.ArgumentParser(description='MS-G3D')

    parser.add_argument(
        '--work-dir',
        type=str,
        required=True,
        help='the work folder for storing results')
    parser.add_argument('--model_saved_name', default='')
    parser.add_argument(
        '--config',
        default='./config/nturgbd-cross-view/test_bone.yaml',
        help='path to the configuration file')
    parser.add_argument(
        '--assume-yes',
        action='store_true',
        help='Say yes to every prompt')

    parser.add_argument(
        '--phase',
        default='train',
        help='must be train or test')
    parser.add_argument(
        '--save-score',
        type=str2bool,
        default=False,
        help='if ture, the classification score will be stored')

    parser.add_argument(
        '--seed',
        type=int,
        default=random.randrange(200),
        help='random seed')
    parser.add_argument(
        '--log-interval',
        type=int,
        default=100,
        help='the interval for printing messages (#iteration)')
    parser.add_argument(
        '--save-interval',
        type=int,
        default=1,
        help='the interval for storing models (#iteration)')
    parser.add_argument(
        '--eval-interval',
        type=int,
        default=1,
        help='the interval for evaluating models (#iteration)')
    parser.add_argument(
        '--eval-start',
        type=int,
        default=1,
        help='The epoch number to start evaluating models')
    parser.add_argument(
        '--print-log',
        type=str2bool,
        default=True,
        help='print logging or not')
    parser.add_argument(
        '--show-topk',
        type=int,
        default=[1, 5],
        nargs='+',
        help='which Top K accuracy will be shown')

    parser.add_argument(
        '--feeder',
        default='feeder.feeder',
        help='data loader will be used')
    parser.add_argument(
        '--num-worker',
        type=int,
        default=os.cpu_count(),
        help='the number of worker for data loader')
    parser.add_argument(
        '--train-feeder-args',
        default=dict(),
        help='the arguments of data loader for training')
    parser.add_argument(
        '--test-feeder-args',
        default=dict(),
        help='the arguments of data loader for test')

    parser.add_argument(
        '--model',
        default=None,
        help='the model will be used')
    parser.add_argument(
        '--model-args',
        type=dict,
        default=dict(),
        help='the arguments of model')
    parser.add_argument(
        '--weights',
        default=None,
        help='the weights for network initialization')
    parser.add_argument(
        '--ignore-weights',
        type=str,
        default=[],
        nargs='+',
        help='the name of weights which will be ignored in the initialization')
    parser.add_argument(
        '--half',
        action='store_true',
        help='Use half-precision (FP16) training')
    parser.add_argument(
        '--amp-opt-level',
        type=int,
        default=1,
        help='NVIDIA Apex AMP optimization level')

    parser.add_argument(
        '--base-lr',
        type=float,
        default=0.01,
        help='initial learning rate')
    parser.add_argument(
        '--step',
        type=int,
        default=[20, 40, 60],
        nargs='+',
        help='the epoch where optimizer reduce the learning rate')
    parser.add_argument(
        '--device',
        type=int,
        default=0,
        nargs='+',
        help='the indexes of GPUs for training or testing')
    parser.add_argument(
        '--optimizer',
        default='SGD',
        help='type of optimizer')
    parser.add_argument(
        '--nesterov',
        type=str2bool,
        default=False,
        help='use nesterov or not')
    parser.add_argument(
        '--batch-size',
        type=int,
        default=32,
        help='training batch size')
    parser.add_argument(
        '--test-batch-size',
        type=int,
        default=256,
        help='test batch size')
    parser.add_argument(
        '--forward-batch-size',
        type=int,
        default=16,
        help='Batch size during forward pass, must be factor of --batch-size')
    parser.add_argument(
        '--start-epoch',
        type=int,
        default=0,
        help='start training from which epoch')
    parser.add_argument(
        '--num-epoch',
        type=int,
        default=80,
        help='stop training in which epoch')
    parser.add_argument(
        '--weight-decay',
        type=float,
        default=0.0005,
        help='weight decay for optimizer')
    parser.add_argument(
        '--optimizer-states',
        type=str,
        help='path of previously saved optimizer states')
    parser.add_argument(
        '--checkpoint',
        type=str,
        help='path of previously saved training checkpoint')
    parser.add_argument(
        '--debug',
        type=str2bool,
        default=False,
        help='Debug mode; default false')
    parser.add_argument(
        '--openpose',
        type=str,
        default='../openpose/build',
        help='openpose path')
    parser.add_argument(
        '--video',
        type=str,
        default='./resource/media/ta_chi.mp4',
        help='openpose path')
    parser.add_argument(
        '--output_dir',
        type=str,
        default='./data/demo_result',
        help='output dir')
    parser.add_argument('--height',
                        default=1080,
                        type=int)
    return parser

colors = [(255, 0, 0), (255, 85, 0), (255, 170, 0), (255, 255, 0), \
                    (170, 255, 0), (85, 255, 0), (0, 255, 0), (0, 255, 85), (0, 255, 170), \
                    (0, 255, 255), (0, 170, 255), (0, 85, 255), (0, 0, 255), (85, 0, 255), \
                    (170, 0, 255), (255, 0, 255), (255, 0, 170), (255, 0, 85) ]
# debug asdadasaaa11111111111111111111
class Processor():
    """Processor for Skeleton-based Action Recgnition"""

    def __init__(self, arg):
        self.arg = arg
        self.load_model()
        self.best_acc = 0
        self.best_acc_epoch = 0
        # from st-gcn
        # self.dev = "cuda:3"
        # for cpu
        # if type(self.arg.device) is list:
        #     if len(self.arg.device) > 1:
        #         self.print_log(f'{len(self.arg.device)} GPUs available, using DataParallel')
        #         self.model = nn.DataParallel(
        #             self.model,
        #             device_ids=self.arg.device,
        #             output_device=self.output_device
        #         )

    def load_model(self):
        output_device = self.arg.device[0] if type(
            self.arg.device) is list else self.arg.device
        self.output_device = output_device
        Model = import_class(self.arg.model)

        # Copy model file and main
        shutil.copy2(inspect.getfile(Model), self.arg.work_dir)
        shutil.copy2(os.path.join('.', __file__), self.arg.work_dir)

        # cpu
        self.model = Model(**self.arg.model_args)
        # gpu
        # self.model = Model(**self.arg.model_args).cuda(output_device)
        # self.loss = nn.CrossEntropyLoss().cuda(output_device)
        self.print_log(f'Model total number of params: {count_params(self.model)}')

    def print_time(self):
        localtime = time.asctime(time.localtime(time.time()))
        self.print_log(f'Local current time: {localtime}')

    def print_log(self, s, print_time=True):
        if print_time:
            localtime = time.asctime(time.localtime(time.time()))
            s = f'[ {localtime} ] {s}'
        print(s)
        if self.arg.print_log:
            with open(os.path.join(self.arg.work_dir, 'log.txt'), 'a') as f:
                print(s, file=f)

    def random_sample_sequence(self, data_numpy, frames):
        C, T, V, M = data_numpy.shape
        length = T // frames
        clips = np.zeros((C, frames, V, M))
        l = 0
        for t in range(frames):
            rand_index = random.randint(l, l + length - 1)
            clips[:, t, :, :] = data_numpy[:, rand_index, :, :]
            l = l + length
        return clips

    def demo2D14(self):

        vid = 'S011C001P025R002A020'
        sample_name = self.data_loader['test'].dataset.sample_name
        sample_id = [name.split('.')[0] for name in sample_name]
        index = sample_id.index(vid)
        data, label, index = self.data_loader['test'].dataset[index]
        data = data.reshape((1,) + data.shape)
        # load openpose python api
        # ipdb.set_trace()11
        video_name = self.arg.video.split('/')[-1].split('.')[0].split('_')[0]
        output_result_dir = self.arg.output_dir
        output_result_path = '{}/{}.mp4'.format(output_result_dir, video_name)
        # skes_path = '../MS-G3D/data/nturgbd_raw/nturgb+d_skeletons/'
        # sks_name = video_name
        # sks_name = 'S001C001P001R001A060'
        # ipdb.set_trace()
        # sks_name = 'S001C001P001R001A032'
        # bodies_data = ntu_skes.get_raw_bodies_data(skes_path, sks_name)
        images = []
        label_name_path = './ntu_label_name.txt'

        with open(label_name_path) as f:
            label_name = f.readlines()
            label_name = [line.rstrip() for line in label_name]
            self.label_name = label_name

        self.model.eval()

        # pack openpose ouputs
        video = utils.video.get_video_frames(self.arg.video)
        height, width, _ = video[0].shape
        multi_pose2D = data[0]
        data = torch.from_numpy(data)
        # data = data.unsqueeze(0)
        data = data.float().cuda(self.output_device).detach()

        # extract feature
        print('\nNetwork forwad...')
        self.model.eval()
        # ipdb.set_trace()
        output, feature = self.model.extract_feature(data)
        output = output[0]
        feature = feature[0]
        intensity = (feature * feature).sum(dim=0) ** 0.5
        intensity = intensity.cpu().detach().numpy()
        # no realtime
        # label = output_normal.argmax(dim = 0)
        label = output.argmax(dim=0)

        print('Prediction result: {}'.format(label_name[label]))
        print('Done.')

        # visualization
        print('\nVisualization...')
        label_name_sequence = None
        graph = 'graph.ntu_rgb_d.Graph'
        G = import_class(graph)()
        edge = G.edge
        # ipdb.set_trace()
        images = utils.visualization1.stgcn_visualize(
            multi_pose2D, edge, intensity, video, label_name[label], label_name_sequence, output, self.arg.height)
        print('Done.')

        # save video
        print('\nSaving...')
        if not os.path.exists(output_result_dir):
            os.makedirs(output_result_dir)
        writer = skvideo.io.FFmpegWriter(output_result_path,
                                         outputdict={'-b': '300000000'})
        for img in images:
            writer.writeFrame(img)
        writer.close()
        print('The Demo result has been saved in {}.'.format(output_result_path))

    def get_rect_by_pts(self, kpt):
        pts_np  = kpt.reshape((-1, 3))
        vis_mask = pts_np[:, 2] > 0
        pts_valid = pts_np[vis_mask]
        minx, miny, _ = pts_valid.min(axis=0)
        maxx, maxy, _ = pts_valid.max(axis=0)
        return [int(minx), int(miny), int(maxx-minx), int(maxy-miny)]

    def draw_human(self, pts, image, box):
        H = 1023
        W = 268
        for i, pt in enumerate(pts):
            x = int(pt[0])
            y = int(pt[1])
            if pt[2] > 1.5:
                continue
            cv2.circle(image, (x, y), 10, colors[i], -1)
        x1, y1 = int(box[0]), int(box[1])
        x2, y2 = x1 + int(box[2]), y1 + int(box[3])
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 0, 255), 3)
        # cv2.putText(image, str(i), (x,y), cv2.FONT_HERSHEY_SIMPLEX, 2.1, (0, 0, 255), 3)
        # scale test video to real video
        image = image[y1:y2, x1:x2]
        h, w, _ = image.shape
        image = cv2.resize(
            image, None, fx=H / h, fy=W / w)
        return image

    def demo_realtime2D14(self):
        # load openpose python api
        hrnet_model = HRNetProcessor().model
        # self.ort_session = ort.InferenceSession('../onnx1/msg3dv214_10_NoRandom.onnx')
        # self.ort_session = ort.InferenceSession('../onnx1/msg3dv214_300.onnx')
        self.ort_session = ort.InferenceSession('../onnx1/msg3dv214_10.onnx')
        # self.ort_session = ort.InferenceSession('./msg3dv214_300.onx')
        #

        # for video
        video_name = self.arg.video.split('/')[-1].split('.')[0].split('_')[0]
        output_result_dir = self.arg.output_dir
        # output_result_path = '{}/{}.mp4'.format(output_result_dir, 'test12')
        output_result_path = '{}/{}.mp4'.format(output_result_dir, 'test22')
        # output_result_path = '{}/{}.mp4'.format(output_result_dir, video_name+'_real_windown10')
        # output_result_path = '{}/{}.mp4'.format(output_result_dir, 'jump_window10_2')
        # output_result_path = '{}/{}.mp4'.format(output_result_dir, 'hopping_window10')
        # output_result_path = '{}/{}.mp4'.format(output_result_dir, 'positive1_30_threshold5_clear')

        # for camer
        # output_result_dir = self.arg.output_dir
        # output_result_path = '{}/{}.mp4'.format(output_result_dir, 'camera')

        images = []
        label_name_path = './ntu_label_name.txt'
        with open(label_name_path) as f:
            label_name = f.readlines()
            label_name = [line.rstrip() for line in label_name]
            self.label_name = label_name

        self.model.eval()
        # pose_tracker2D = realtime.naive_pose_tracker(data_frame=300, num_joint=14)
        # pose_tracker2D = realtime.naive_pose_tracker(data_frame=300, num_joint=14)
        # pose_tracker2D = realtime.naive_pose_tracker(data_frame=300, num_joint=14)
        pose_tracker2D = realtime.naive_pose_tracker(data_frame=300, num_joint=14)
        if self.arg.video == 'camera_source':
            video_capture = cv2.VideoCapture(0)
        else:
            video_capture = cv2.VideoCapture(self.arg.video)

        # start recognition
        start_time = time.time()
        frame_index = 0
        sum1 = 0
        sum2 = 0
        if not os.path.exists(output_result_dir):
            os.makedirs(output_result_dir)

        writer = skvideo.io.FFmpegWriter(output_result_path,
                                         outputdict={'-r': '30'})
        graph = 'graph.ntu_rgb_d.Graph'
        G = import_class(graph)()
        hold_on = 0
        holds_on = {'falling down': 3, 'jump up': 2, 'put on jacket': 4, 'stand up': 2, 'cheer up': 2, 'kicking something': 2, 'take off jacket': 4, 'hopping': 2, 'sit down': 2, 'staggering': 2}
        # flag = 1
        flag = 0
        edge = G.edge
        voting_label_name = 'others'
        confidence1 = 1
        count = 0
        box_list = []
        dw = 0
        dh = 0
        H = 1023
        W = 268
        while (True):
            tic = time.time()
            # ipdb.set_trace()
            # get image
            ret, orig_image = video_capture.read()
            # orig_image = cv2.flip(orig_image, 90)
            # ipdb.set_trace()
            if orig_image is None:
                break
            source_H, source_W, _ = orig_image.shape
            # fix phone camer bug
            # orig_image = np.rot90(orig_image, -1)
            # orig_image = cv2.UMat(orig_image).get()  # np convert to cv2 again, fix bugs
            # source_H, source_W, _ = orig_image.shape
            # hrnet predict
            # scale test video
            time1 = time.time()
            res_joint = hrnet_model.predict(orig_image)
            time2 = time.time()
            print("hrnet1 time:{}".format(time2 - time1))
            multi_pose2D = np.zeros((1, 14, 3))  # (num_person, num_joint, 3)
            max_box_areas = 0
            keypoints = []
            for body, pts in enumerate(res_joint):
                # only caulate max box
                kpt, num_keypoints = convert_to_universal(pts)
                box = self.get_rect_by_pts(kpt)
                box_areas = box[2] * box[3]
                if box_areas > max_box_areas:
                    # ipdb.set_trace()
                    max_box_areas = box_areas
                    keypoints = kpt.reshape(-1).tolist()
                    max_box = box
            # ipdb.set_trace()
            # dw += max_box[2]
            # dh += max_box[3]
            # box_list.append((max_box[2], max_box[3]))
            # ipdb.set_trace()
            # image = self.draw_human(max_kpt, orig_image, max_box)
            # save_name = './vis1/%03d.jpg' % frame_index
            # cv2.imwrite(save_name, image)
            # '''scal test frames to real frames
            #    get new keypoints
            # '''
            # x1, y1 = int(max_box[0]), int(max_box[1])
            # x2, y2 = x1 + int(max_box[2]), y1 + int(max_box[3])
            # orig_image = orig_image[y1:y2, x1:x2]
            # h, w, _ = orig_image.shape
            # orig_image = cv2.resize(
            #     orig_image, None, fx=H / h, fy=W / w)
            # time1 = time.time()
            # res_joint = hrnet_model.predict(orig_image)
            # time2 = time.time()
            # print("hrnet2 time:{}".format(time2 - time1))
            # for body, pts in enumerate(res_joint):
            #     kpt, num_keypoints = convert_to_universal(pts)
            #     keypoints = kpt.reshape(-1).tolist()
            # frame_index += 1
            # # ipdb.set_trace()
            j = 0
            for i in range(0, len(keypoints), 3):
                multi_pose2D[0][j][0] = keypoints[i] / source_W
                multi_pose2D[0][j][1] = keypoints[i+1] / source_H
                multi_pose2D[0][j][2] = keypoints[i+2]
                j += 1
                # keypoints = kpt.reshape(-1).tolist()
                # j = 0
                # for i in range(0, len(keypoints), 3):
                #     multi_pose2D[body][j][0] = keypoints[i] / source_W
                #     multi_pose2D[body][j][1] = keypoints[i+1] / source_H
                #     multi_pose2D[body][j][2] = keypoints[i+2]
                #     j += 1
            orig_image = cv2.resize(
                orig_image, (256 * source_W // source_H, 256))
            H, W, _ = orig_image.shape
            multi_pose2D[:, :, 0:2] = multi_pose2D[:, :, 0:2] - 0.5
            multi_pose2D[:, :, 0][multi_pose2D[:, :, 2] == 0] = 0
            multi_pose2D[:, :, 1][multi_pose2D[:, :, 2] == 0] = 0

            # ipdb.set_trace()
            # pose tracking
            if self.arg.video == 'camera_source':
                frame_index = int((time.time() - start_time) * self.arg.fps)
            else:
                frame_index += 1
            pose_tracker2D.update(multi_pose2D, frame_index)
            # pose_tracker2D1.update(multi_pose2D, frame_index)
            # pose_tracker2D2.update(multi_pose2D, frame_index)

            data_numpy2D = pose_tracker2D.get_skeleton_sequence()

            # data_numpy2D1 = pose_tracker2D1.get_skeleton_sequence()
            # data_numpy2D2 = pose_tracker2D2.get_skeleton_sequence()

            # fix bug, only pad 300 frames for data_numpy not for data_numpy2D
            data_tmp = copy.deepcopy(data_numpy2D)

            # data_tmp1 = copy.deepcopy(data_numpy2D1)
            # data_tmp2 = copy.deepcopy(data_numpy2D2)

            data_numpy = np.expand_dims(data_tmp, axis=0)
            # data_numpy1 = np.expand_dims(data_tmp1, axis=0)
            # data_numpy2 = np.expand_dims(data_tmp2, axis=0)

            data_numpy = pad_2D(data_numpy)
            # data_numpy1 = pad_2D(data_numpy)
            # data_numpy2 = pad_2D(data_numpy)

            # data_numpy = pre_normalization(data)
            # randomSample
            data_numpy = self.random_sample_sequence(data_numpy[0], 10)
            # data_numpy1 = self.random_sample_sequence(data_numpy1[0], 10)
            # data_numpy2 = self.random_sample_sequence(data_numpy2[0], 10)

            data_numpy = np.expand_dims(data_numpy, axis=0)
            # data_numpy1 = np.expand_dims(data_numpy1, axis=0)
            # data_numpy2 = np.expand_dims(data_numpy2, axis=0)

            # data = torch.from_numpy(tmp)
            # debug
            # data = torch.from_numpy(data_numpy)

            # data = torch.from_numpy(data_numpy)
            # data = data.unsqueeze(0)
            # gpu
            # data = data.float().cuda(self.output_device).detach()
            # cpu
            # data = data.float()
            # ipdb.set_trace()
            # if flag == 1:
            voting_label_name, _, confidence1 = self.predict(data_numpy)
            # ipdb.set_trace()
            # ipdb.set_trace()
            # if voting_label_name != 'others':
            #     flag = 1
            # if voting_label_name == 'others' and flag == 1:
            #     count += 1
            #     if count == 5:
            #         pose_tracker2D.clear()
            #         count = 0
            #         flag = 0
            # if voting_label_name != 'others':
            #     flag = 0
            #     hold_on += 1
            #     threshold = holds_on[voting_label_name]
            #     if hold_on >= threshold*30:
            #         hold_on = 0
            #         flag = 1
            #         pose_tracker2D.clear()
            # if voting_label_name == 'others':
            #     hold
            # voting_label_name1, _, confidence2 = self.predict(data_numpy1)
            # voting_label_name2, _, confidence3 = self.predict(data_numpy2)
            # voting_label_names = [voting_label_name, voting_label_name1, voting_label_name2]
            # confidences = [confidence1, confidence2, confidence3]
            # index = confidences.index(max(confidences))
            # voting_label_res = voting_label_names[index]
            # voting_label_name, output, intensity = self.predict(data)
            # visualization
            app_fps = 1 / (time.time() - tic)
            # ipdb.set_trace()
            image = realtime.DemoRealtime().render(data_numpy2D, data_numpy[0], frame_index, voting_label_name, confidence1,
                                                   orig_image, edge, app_fps)
            # image = realtime.DemoRealtime().render(data_numpy2D, data_numpy[0], voting_label_res, confidences[index],
            #                                        orig_image, edge, app_fps)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            writer.writeFrame(image)
            # writer.write(image)
        writer.close()
        print("save" + output_result_path)
        # writer.release()
    def predict(self, data):
        # torch.cuda.synchronize()
        time1 = time.time()
        # output = self.model(data)
        output = self.ort_session.run(None, {'actual_input_1': data.astype(np.float32)})
        # output = self.model.forward(data)
        # torch.cuda.synchronize()
        time2 = time.time()
        print("inference time:{}".format(time2-time1))

        # output, feature = self.model.extract_feature(data)
        output = output[0][0]
        # feature = feature[0]
        # intensity = (feature * feature).sum(dim=0) ** 0.5
        # intensity = intensity.cpu().detach().numpy()
        # voting_label = output.argmax(dim=0)
        voting_label = output.argmax()
        confidence = round(output[voting_label], 2)
        voting_label_name = self.label_name[voting_label]
        # if confidence < 6 :
        #     voting_label_name = self.label_name[10]
        # else:
        #     voting_label_name = self.label_name[voting_label]
        # return voting_label_name, output, intensity
        return voting_label_name, output, confidence


    def start(self):
        if self.arg.phase == 'demo_realtime2D14':
            if not self.arg.test_feeder_args['debug']:
                wf = os.path.join(self.arg.work_dir, 'wrong-samples.txt')
                rf = os.path.join(self.arg.work_dir, 'right-samples.txt')
            else:
                wf = rf = None
            if self.arg.weights is None:
                raise ValueError('Please appoint --weights.')

            self.print_log(f'Model:   {self.arg.model}')
            self.print_log(f'Weights: {self.arg.weights}')
            self.demo_realtime2D14()
            self.print_log('Done.\n')


def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def main():
    parser = get_parser()

    # load arg form config file
    p = parser.parse_args()
    if p.config is not None:
        with open(p.config, 'r') as f:
            default_arg = yaml.load(f)
        key = vars(p).keys()
        for k in default_arg.keys():
            if k not in key:
                print('WRONG ARG:', k)
                assert (k in key)
        parser.set_defaults(**default_arg)

    arg = parser.parse_args()
    init_seed(arg.seed)
    # ipdb.set_trace()
    processor = Processor(arg)
    processor.start()


if __name__ == '__main__':
    main()