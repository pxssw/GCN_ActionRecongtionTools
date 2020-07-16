# train = os.listdir('/6T-4/fengyu/output/ntu_train')
# val = os.listdir('/6T-4/fengyu/output/ntu_val')
# total = train + val
# videos = os.listdir('/6T-4/fengyu/dataset')
# videos_set = set()
# files_set = set()
# for video in videos:
#     video = video.split('.')[0].split('_')[0]
#     videos_set.add(video)
# for file in total:
#     file = file.split('.')[0]
#     files_set.add(file)
#
# missing = videos_set.symmetric_difference(files_set)
# print(len(missing))
# missing_file = './missing.txt'
# f = open(missing_file, 'w')
# f.writelines(str(missing))
# f.close()

import os
import cv2
import ipdb
from tqdm import tqdm

if __name__ == '__main__':
    # prefix = '/6T-3/fengyu/'
    # datatype = 'dataset'
    # video_paths = []
    # video_prefix  = os.path.join(prefix, datatype)
    # for root_dir, sub_dirs, filenames in os.walk(video_prefix):
    #     for filename in filenames:
    #         if filename.endswith('txt'):
    #             continue
    #         video_path = os.path.join(root_dir, filename)
    #         video_paths.append(video_path)

    # ipdb.set_trace()#11111111
    dun1 = '/6T-4/fengyu/dun3'
    dun2 = '/6T-4/fengyu/dun2'
    logs = './logs'
    if not os.path.exists(dun1):
        os.makedirs(dun1)
    if not os.path.exists(dun2):
        os.makedirs(dun2)
    root = '/6T-4/fengyu/dataset/'
    # for idx ,miss in tqdm(enumerate(missing)):
    images = []
    video_name = 'S016C003P008R001A012_rgb.avi'
    video_path = os.path.join(root, video_name)
    cap = cv2.VideoCapture(video_path)
    video_name = video_path.split('/')[-1].split('.')[0].split('_')[0]
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        images.append(frame)
    print("get {} images in {}".format(len(images), video_name))
    img_prefix = os.path.join(dun2, video_name)
    if not os.path.exists(img_prefix):
        os.makedirs(img_prefix)
    for img_id, image in enumerate(images):
        img_name =video_name + '_' +"%06d"%img_id + '.jpg'
        img_path = os.path.join(img_prefix, img_name)
        # ipdb.set_trace()
        # if os.path.exists(img_path):
        #     continue
        cv2.imwrite(img_path, image)