import cv2
import torch
from SimpleHRNet import SimpleHRNet
import json
import os
import numpy as np

colors = [(255, 0, 0), (255, 85, 0), (255, 170, 0), (255, 255, 0), \
                    (170, 255, 0), (85, 255, 0), (0, 255, 0), (0, 255, 85), (0, 255, 170), \
                    (0, 255, 255), (0, 170, 255), (0, 85, 255), (0, 0, 255), (85, 0, 255), \
                    (170, 0, 255), (255, 0, 255), (255, 0, 170), (255, 0, 85) ]
def draw_human(pts, image, box):
    for i, pt in enumerate(pts):
        x = int(pt[0])
        y = int(pt[1])
        if pt[2] > 1.5:
            continue
        cv2.circle(image, (x, y), 10, colors[i], -1)
        x1, y1 = int(box[0]), int(box[1])
        x2, y2 = x1 + int(box[2]), y1 + int(box[3])
        cv2.rectangle(image, (x1, y1) , (x2, y2),  (0, 0, 255), 3 )
        #cv2.putText(image, str(i), (x,y), cv2.FONT_HERSHEY_SIMPLEX, 2.1, (0, 0, 255), 3)
    return image

def convert_to_universal(pts, thres=0.1):
    #mpii to universal
    idx = [9,8,12, 11, 10,13,14,15, 2, 1, 0, 3, 4, 5]
    kpt = pts[idx]
    num_keypoints = 0
    for j in range(len(kpt)):
        if kpt[j, 2] < thres:
            kpt[j, 2] = 0
        else:
            kpt[j, 2] = 1
            num_keypoints += 1
    kpt = kpt[:, [1, 0,2]]
    return kpt, num_keypoints


import ipdb
class HRNetProcessor(object):
    def __init__(self):
        self.model = SimpleHRNet(32, 16, "simpleHRNet/weights/mpii/pose_hrnet_w32_256x256.pth", \
            resolution=(256, 256), multiperson=True, device=torch.device('cuda:0'))
        # self.model = SimpleHRNet(32, 16, "simpleHRNet/weights/mpii/pose_hrnet_w32_256x256.pth", \
        #     resolution=(256, 256), multiperson=True)
if __name__ == '__main__':
    prefix = '/home/ubuntu/file/raid5/space/chaidao/download_videos/'
    datatype = 'dun'
    hr_processer = HRNetProcessor(prefix, datatype)
    # hr_processer.convert_to_universal()
    # hr_processer.process()
    # hr_processer.dump_coco_json('./dun.json')


