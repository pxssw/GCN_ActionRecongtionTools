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


class HRNetProcessor(object):
    def __init__(self, prefix, datatype):
        self.prefix = prefix
        self. datatype = datatype
        self.model = SimpleHRNet(32, 16, "./weights/mpii/pose_hrnet_w32_256x256.pth", \
            resolution=(256, 256), multiperson=True, device=torch.device('cuda:0'))
        self.coco = {'images':[], 'categories':[], 'annotations':[]}
        # self.coco = {'pe':[]}

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
        image_paths = self.get_image_paths(video_prefix)
        print('we have {} images to be processed'.format(len(image_paths)))
        a_id = 0
        for img_id, image_path  in enumerate(image_paths):
            image = cv2.imread(image_path, 1) 
            width, height = image.shape[1], image.shape[0]
            res_joint = self.model.predict(image)
            relate_path = image_path.replace(self.prefix, '')
            img_dict = {'id': img_id, 'file_name':relate_path, 'width':width, 'height':height}
            self.coco['images'].append(img_dict)
            if img_id %100 == 0:
                print('process {} /{}'.format(img_id, len(image_paths)))
            if len(res_joint) < 1:
                continue
            for pts in res_joint:
                kpt, num_keypoints = self.convert_to_universal(pts)
                if num_keypoints < 5:
                    continue
                box = self.get_rect_by_pts(kpt)
                person_dict = {'id':a_id, 'image_id':img_id, 'category_id':1, \
                    'area':int(box[2]*box[3]), 'bbox':box, 'iscrowd':False,\
                    'keypoints':kpt.reshape(-1).tolist(), 'num_keypoints':num_keypoints}                
                self.coco['annotations'].append(person_dict)
                a_id += 1
                #image = draw_human(kpt, image, box)
            #save_name = './vis/%03d.jpg'%img_id
            #cv2.imwrite(save_name, image)
        len_im = len(self.coco['images'])
        len_ann = len(self.coco['annotations']) 
        print('have {} images, {} annos'.format(len_im, len_ann) )
        print('last img_id {}, last ann_id:{}'.format(self.coco['images'][-1]['id'], \
            self.coco['annotations'][-1]['id']))
        self.coco['categories']=[self.category]
    

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

    def get_image_paths(self, video_prefix):
        image_paths = []
        for root_dir, sub_dirs, filenames in os.walk(video_prefix):
            for filename in filenames:
                if filename.endswith('.jpg'):
                    image_paths.append(os.path.join(root_dir, filename))
        return image_paths

    def dump_coco_json(self, json_name):
        with open(json_name, 'w') as f:
            json.dump(self.coco, f)

if __name__ == '__main__':
    prefix = '/6T-4/fengyu/'
    datatype = 'dun2'
    hr_processer = HRNetProcessor(prefix, datatype)
    hr_processer.process()
    hr_processer.dump_coco_json('./dun.json')


