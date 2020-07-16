#!/usr/bin/env python
import os
import sys
import argparse
import json
import shutil
import time

import numpy as np
import torch
import skvideo.io

import tools
import tools.utils as utils

import cv2
import ipdb
class DemoRealtime():
    """ A demo for utilizing st-gcn in the realtime action recognition.
    The Openpose python-api is required for this demo.

    Since the pre-trained model is trained on videos with 30fps,
    and Openpose is hard to achieve this high speed in the single GPU,
    if you want to predict actions by **camera** in realtime,
    either data interpolation or new pre-trained model
    is required.

    Pull requests are always welcome.
    """
    def render(self, data_numpy, sample_data, frame_index, voting_label_name, confidence, orig_image, edge, fps=0):
        images = utils.visualization.stgcn_visualize(
            frame_index,
            data_numpy[:, [-1]],
            sample_data,
            edge,
            [orig_image],
            voting_label_name,
            confidence,
            1080,
            fps=fps)
        image = next(images)
        image = image.astype(np.uint8)
        return image


class naive_pose_tracker():
    """ A simple tracker for recording person poses and generating skeleton sequences.
    For actual occasion, I recommend you to implement a robuster tracker.
    Pull-requests are welcomed.
    """
    def __init__(self, data_frame=300, num_joint=25):
        self.data_frame = data_frame
        self.num_joint = num_joint
        self.latest_frame = 0
        self.trace_info = list()

    def update(self, multi_pose, current_frame):

        if current_frame <= self.latest_frame:
            return
        if len(multi_pose.shape) != 3:
            return
        # if current_frame >= 5:
        #     ipdb.set_trace()
        matching_trace = None
        for trace_index, (trace, latest_frame) in enumerate(self.trace_info):
            # trace.shape: (num_frame, num_joint, 3)
            if current_frame <= latest_frame:
                continue
            matching_trace = trace_index

        # update trace information
        if matching_trace is not None:
            trace, latest_frame = self.trace_info[matching_trace]
            new_trace = np.concatenate((trace, [multi_pose[0]]), 0)
            self.trace_info[matching_trace] = (new_trace, current_frame)
        else:
            new_trace = np.array([multi_pose[0]])
            self.trace_info.append((new_trace, current_frame))

        self.latest_frame = current_frame

    def clear(self):
        self.trace_info.clear()


    def get_skeleton_sequence(self):
        num_trace = len(self.trace_info)
        if num_trace == 0:
            return None
        data = np.zeros((3, self.data_frame, self.num_joint, num_trace))
        for trace_index, (trace, latest_frame) in enumerate(self.trace_info):
            # end = self.data_frame - (self.latest_frame - latest_frame)
            end = self.data_frame
            d = trace[-end:]
            beg = end - len(d)
            data[:, beg:end, :, trace_index] = d.transpose((2, 0, 1))
        return data[:,:,:,0:2]
