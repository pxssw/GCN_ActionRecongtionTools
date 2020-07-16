#!/usr/bin/env python
from __future__ import print_function
import os
import time
import yaml
import random
import argparse
import torch
import ipdb
import tools.demo_realtime as realtime
import skvideo.io
import sys
import cv2
from utils import import_class
sys.path.append('./simpleHRNet')
from simpleHRNet.processbyhrnet0 import convert_to_universal
from simpleHRNet.processbyhrnet0 import HRNetProcessor
from tools.preprocess import pad_2D
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
    parser = argparse.ArgumentParser(description='Demo')
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
    parser.add_argument(
        '--seed',
        type=int,
        default=random.randrange(200),
        help='random seed')
    parser.add_argument('--height',
                        default=1080,
                        type=int)
    return parser

colors = [(255, 0, 0), (255, 85, 0), (255, 170, 0), (255, 255, 0), \
                    (170, 255, 0), (85, 255, 0), (0, 255, 0), (0, 255, 85), (0, 255, 170), \
                    (0, 255, 255), (0, 170, 255), (0, 85, 255), (0, 0, 255), (85, 0, 255), \
                    (170, 0, 255), (255, 0, 255), (255, 0, 170), (255, 0, 85) ]
class Processor():
    """Processor for Skeleton-based Action Recgnition"""

    def __init__(self, arg):
        self.arg = arg

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

    def get_rect_by_pts(self, kpt):
        pts_np  = kpt.reshape((-1, 3))
        vis_mask = pts_np[:, 2] > 0
        pts_valid = pts_np[vis_mask]
        minx, miny, _ = pts_valid.min(axis=0)
        maxx, maxy, _ = pts_valid.max(axis=0)
        return [int(minx), int(miny), int(maxx-minx), int(maxy-miny)]

    def scale(self, image, box):
        scale_h = 0.39
        scale_w = 0.059
        x1, y1 = box[0], box[1]
        x2, y2 = x1 + box[2], y1 + box[3]
        # scale test video to real video
        center_x = int((x1 + x2)/2)
        center_y = int((y1 + y2)/2)
        dx = int(box[2] / scale_w)
        dy = int(box[3] / scale_h)
        new_x = center_x - dx//2
        new_y = center_y - dy//2
        new_x1 = new_x + dx
        new_y1 = new_y + dy
        image = image[new_y:new_y1, new_x:new_x1]
        if image.shape[0] == 0 or image.shape[1] == 0:
            ipdb.set_trace()
        image = cv2.resize(image, (1920, 1080))
        return image

    def draw_human(self, pts, image, box):
        for i, pt in enumerate(pts):
            x = int(pt[0])
            y = int(pt[1])
            if pt[2] > 1.5:
                continue
            cv2.circle(image, (x, y), 10, colors[i], -1)
        x1, y1 = box[0], box[1]
        x2, y2 = x1 + box[2], y1 + box[3]
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 0, 255), 3)
        return image

    def pad(self, img):
        BLACK = [0, 0, 0]
        H, W, _ = img.shape
        h = int(H * 3)
        w = int(W * 5)
        top = (h - H) // 2
        bottom = (w - W) // 2
        if top + bottom + H < h:
            bottom += 1

        left = (w - W) // 2
        right = (w - W) // 2
        if left + right + W < w:
            right += 1
        pad_image = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=BLACK)
        return pad_image

    def demo_realtime2D14(self):
        # load openpose python api
        hrnet_model = HRNetProcessor().model
        self.ort_session = ort.InferenceSession('./onnx_models/msg3dv214_10.onnx')
        # for video
        video_name = self.arg.video.split('/')[-1].split('.')[0].split('_')[0]
        output_result_dir = self.arg.output_dir
        output_result_path = '{}/{}.mp4'.format(output_result_dir, video_name)
        # output_result_path = '{}/{}.mp4'.format(output_result_dir, 'test25')

        # for camer
        # output_result_dir = self.arg.output_dir
        # output_result_path = '{}/{}.mp4'.format(output_result_dir, 'camera')

        images = []
        label_name_path = './ntu_label_name.txt'
        with open(label_name_path) as f:
            label_name = f.readlines()
            label_name = [line.rstrip() for line in label_name]
            self.label_name = label_name
        pose_tracker2D = realtime.naive_pose_tracker(data_frame=300, num_joint=14)
        if self.arg.video == 'camera_source':
            video_capture = cv2.VideoCapture(0)
        else:
            video_capture = cv2.VideoCapture(self.arg.video)
        # start recognition
        start_time = time.time()
        frame_index = 0
        if not os.path.exists(output_result_dir):
            os.makedirs(output_result_dir)
        writer = skvideo.io.FFmpegWriter(output_result_path,
                                         outputdict={'-r': '30'})
        graph = 'graph.ntu_rgb_d.Graph'
        G = import_class(graph)()
        edge = G.edge
        first_max_box = []
        while (True):
            tic = time.time()
            # get image
            ret, orig_image = video_capture.read()
            # ipdb.set_trace()
            if orig_image is None:
                break
            # source_H, source_W, _ = orig_image.shape
            # fix phone camer bug
            orig_image = np.rot90(orig_image, -1)
            orig_image = cv2.UMat(orig_image).get()  # np convert to cv2 again, fix bugs
            orig_image = self.pad(orig_image)
            # hrnet predict
            # scale test video
            multi_pose2D = np.zeros((1, 14, 3))  # (num_person, num_joint, 3)
            max_box_areas = 0
            keypoints = []
            # using first frame for center point
            # ----------------------------------------------#
            if frame_index == 0:
                time1 = time.time()
                res_joint = hrnet_model.predict(orig_image)
                time2 = time.time()
                print("hrnet1 time:{}".format(time2 - time1))
                for body, pts in enumerate(res_joint):
                    # only caulate max box
                    kpt, num_keypoints = convert_to_universal(pts)
                    box = self.get_rect_by_pts(kpt)
                    box_areas = box[2] * box[3]
                    if box_areas > max_box_areas:
                        # ipdb.set_trace()
                        max_box_areas = box_areas
                        max_kpt = kpt
                        keypoints = kpt.reshape(-1).tolist()
                        first_max_box = box
            orig_image = self.scale(orig_image, first_max_box)
            # ----------------------------------------------#
            '''detecting scaled frames
               get new keypoints
            '''
            # ----------------------------------------------#
            time1 = time.time()
            res_joint = hrnet_model.predict(orig_image)
            time2 = time.time()
            print("hrnet2 time:{}".format(time2 - time1))
            for body, pts in enumerate(res_joint):
                kpt, num_keypoints = convert_to_universal(pts)
                keypoints = kpt.reshape(-1).tolist()
            frame_index += 1
            # ipdb.set_trace()
            source_H, source_W, _ = orig_image.shape
            j = 0
            for i in range(0, len(keypoints), 3):
                multi_pose2D[0][j][0] = keypoints[i] / source_W
                multi_pose2D[0][j][1] = keypoints[i+1] / source_H
                multi_pose2D[0][j][2] = keypoints[i+2]
                j += 1
            # ----------------------------------------------#
            orig_image = cv2.resize(
                orig_image, (256 * source_W // source_H, 256))
            multi_pose2D[:, :, 0:2] = multi_pose2D[:, :, 0:2] - 0.5
            multi_pose2D[:, :, 0][multi_pose2D[:, :, 2] == 0] = 0
            multi_pose2D[:, :, 1][multi_pose2D[:, :, 2] == 0] = 0
            # pose tracking
            if self.arg.video == 'camera_source':
                frame_index = int((time.time() - start_time) * self.arg.fps)
            else:
                frame_index += 1
            pose_tracker2D.update(multi_pose2D, frame_index)
            data_numpy2D = pose_tracker2D.get_skeleton_sequence()

            # fix bug, only pad 300 frames for data_numpy not for data_numpy2D
            data_tmp = copy.deepcopy(data_numpy2D)
            data_numpy = np.expand_dims(data_tmp, axis=0)
            data_numpy = pad_2D(data_numpy)
            # randomSample
            data_numpy = self.random_sample_sequence(data_numpy[0], 10)
            data_numpy = np.expand_dims(data_numpy, axis=0)
            voting_label_name, _, confidence1 = self.predict(data_numpy)
            app_fps = 1 / (time.time() - tic)
            # ipdb.set_trace()
            image = realtime.DemoRealtime().render(data_numpy2D, data_numpy[0], frame_index, voting_label_name, confidence1,
                                                   orig_image, edge, app_fps)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            writer.writeFrame(image)
            # writer.write(image)
        writer.close()
        print("save" + output_result_path)
    def predict(self, data):
        # torch.cuda.synchronize()
        time1 = time.time()
        # output = self.model(data)
        output = self.ort_session.run(None, {'actual_input_1': data.astype(np.float32)})
        # output = self.model.forward(data)
        # torch.cuda.synchronize()
        time2 = time.time()
        print("inference time:{}".format(time2-time1))
        output = output[0][0]
        voting_label = output.argmax()
        confidence = round(output[voting_label], 2)
        voting_label_name = self.label_name[voting_label]
        return voting_label_name, output, confidence


    def start(self):
        self.demo_realtime2D14()


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
    arg = parser.parse_args()
    init_seed(arg.seed)
    # ipdb.set_trace()
    processor = Processor(arg)
    processor.start()


if __name__ == '__main__':
    main()