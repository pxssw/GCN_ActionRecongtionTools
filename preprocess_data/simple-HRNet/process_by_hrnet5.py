import cv2
import torch
from SimpleHRNet import SimpleHRNet
import json
import os
import numpy as np
import ipdb
from pathlib import Path
from tqdm import tqdm

# os.environ["CUDA_VISIBLE_DEVICES"] = "2"

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


class HRNetProcessor(object):
    def __init__(self, prefix, datatype):
        self.prefix = prefix
        self. datatype = datatype
        # self.model = SimpleHRNet(32, 16, "./weights/mpii/pose_hrnet_w32_256x256.pth", \
        #     resolution=(256, 256), multiperson=True, device=torch.device('cuda:0'))
        self.model = SimpleHRNet(32, 16, "./weights/mpii/pose_hrnet_w32_256x256.pth", \
            resolution=(256, 256), multiperson=True, device=torch.device('cuda'))
        # self.coco = {'images':[], 'categories':[], 'annotations':[]}
        self.coco = {'people':[]}
        self.category={
            'supercategory':'person',
            'id':1,
            'name':'person',
            'skeleton':[
                [0,1], # head_top, upper_neck
                [1,2], # upper_neck, r_shoulder
                [2,3], # r_shoulder, r_elbow
                [3,4], # r_elbow， r_wrist
                [1,5], # upper_neck, l_shoulder
                [5,6], # l_shoulder， l_elbow
                [6,7], # l_elbow， l_wrist
                [8,9], # r_hip， r_knee
                [9,10], # r_knee， r_ankle
                [11,12], # l_hip， l_knee
                [12,13], # l_knee， l_ankle
            ],
            'keypoints':[
                "head_top", "upper_neck", 
                "r_shoulder", "r_elbow", "r_wrist", 
                "l_shoulder", "l_elbow", "l_wrist",
                "r_hip", "r_knee", "r_ankle",
                "l_hip", "l_knee", "l_ankle"
            ]
        }

    def process(self):
        video_prefix = os.path.join(self.prefix, self.datatype)
        video_paths = os.listdir(video_prefix)
        train_total_labels = dict()
        val_total_labels = dict()
        # output_label_path = '/6T-3/fengyu/output/label'
        # output_label_path = '/6T-4/fengyu/output/label'
        print('we have {} videos to be processed'.format(len(video_paths)))
        for video_id, video_path  in tqdm(enumerate(video_paths)):
            # if video_id % 100 == 0:
            print('process {} /{}'.format(video_id, len(video_paths)))
            video_root = os.path.join(video_prefix, video_path)
            image_files = os.listdir(video_root)
            tmp = int(video_path.split('C')[1].split('P')[0])
            if tmp == 2 or tmp == 3:
                output_sequence_dir = '/6T-4/fengyu/output/ntu_train'

            elif tmp == 1:
                output_sequence_dir = '/6T-4/fengyu/output/ntu_val'

            output_sequence_path = '{}/{}.json'.format(output_sequence_dir, video_path)
            # debug
            if os.path.exists(output_sequence_path):
                # ipdb.set_trace()
                print('exists {}'.format(video_path))
                continue
            if not os.path.exists(output_sequence_dir):
                os.makedirs(output_sequence_dir)
            sequence_info = []
            for idx, image_file in enumerate(sorted(image_files)):
                image_path = os.path.join(video_root, image_file)
                image = cv2.imread(image_path, 1)
                if idx == 0:
                    width, height = image.shape[1], image.shape[0]
                res_joint = self.model.predict(image)
                if len(res_joint) < 1:
                    continue
                frame_data = {'frame_index': int(image_file.split('.')[0].split('_')[-1])}
                skeletons = []
                for pts in res_joint:
                    kpt, num_keypoints = self.convert_to_universal(pts)
                    if num_keypoints < 5:
                        continue
                    skeleton = {}
                    score, coordinates = [], []
                    # box = self.get_rect_by_pts(kpt)
                    # person_dict = {'pose_keypoints_2d':kpt.reshape(-1).tolist()}
                    keypoints = kpt.reshape(-1).tolist()
                    for i in range(0, len(keypoints), 3):
                        coordinates += [keypoints[i] / width, keypoints[i + 1] / height]
                        score += [keypoints[i + 2]]
                    skeleton['pose'] = coordinates
                    skeleton['score'] = score
                    skeletons += [skeleton]
                frame_data['skeleton'] = skeletons
                sequence_info += [frame_data]
            video_info = dict()
            video_info['data'] = sequence_info
            video_info['label'] = 'unknown'
            video_info['label_index'] = int(video_path.split('A')[1]) - 1
            with open(output_sequence_path, 'w') as outfile:
                json.dump(video_info, outfile)
            if len(video_info['data']) == 0:
                print('Can not find pose estimation results.')
                return
            else:
                print('Pose {} estimation complete.'.format(video_id))
            # video = {"has_skeleton": True}
            # video["label_index"] = int(video_path.split('A')[1]) - 1
            # # video["index"] = labels[filename]
            # video["index"] = video_id
            # if tmp == 2 or tmp == 3:
            #     train_total_labels[video_path] = video
            # elif tmp == 1:
            #     val_total_labels[video_path] = video
        print('generate label complete.')


    def get_rect_by_pts(self, kpt):
        pts_np  = kpt.reshape((-1, 3))
        vis_mask = pts_np[:, 2] > 0 
        pts_valid = pts_np[vis_mask]
        minx, miny, _ = pts_valid.min(axis=0)
        maxx, maxy, _ = pts_valid.max(axis=0)
        return [int(minx), int(miny), int(maxx-minx), int(maxy-miny)]

    def convert_to_universal(self, pts, thres=0.1):
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

    def dump_coco_json(self, coco, json_name):
        with open(json_name, 'w') as f:
            json.dump(coco, f, indent=4)

    # def json_pack(self, snippets_dir, video_name, frame_width, frame_height, label='unknown', label_index=-1):
    #     sequence_info = []
    #     p = Path(snippets_dir)
    #     for path in sorted(p.glob(video_name + '*.json')):
    #         # ipdb.set_trace()
    #         json_path = str(path)
    #         # print(path)
    #         frame_id = int(path.stem.split('_')[-1])
    #         frame_data = {'frame_index': frame_id}
    #         data = json.load(open(json_path))
    #         skeletons = []
    #         for person in data['people']:
    #             score, coordinates = [], []
    #             skeleton = {}
    #             keypoints = person['pose_keypoints_2d']
    #             # ipdb.set_trace()
    #             # normalization
    #             for i in range(0, len(keypoints), 3):
    #                 coordinates += [keypoints[i] / frame_width, keypoints[i + 1] / frame_height]
    #                 score += [keypoints[i + 2]]
    #             skeleton['pose'] = coordinates
    #             skeleton['score'] = score
    #             skeletons += [skeleton]
    #         frame_data['skeleton'] = skeletons
    #         sequence_info += [frame_data]
    #
    #     video_info = dict()
    #     video_info['data'] = sequence_info
    #     video_info['label'] = label
    #     video_info['label_index'] = label_index
    #
    #     return video_info

if __name__ == '__main__':
    prefix = '/6T-4/fengyu/'
    datatype = 'dun5'
    hr_processer = HRNetProcessor(prefix, datatype)
    hr_processer.process()
    # hr_processer.dump_coco_json('./dun.json')


