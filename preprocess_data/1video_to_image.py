import os
import cv2
import ipdb
from tqdm import tqdm

if __name__ == '__main__':
    prefix = '/6T-4/fengyu/'
    datatype = 'dataset'
    video_paths = []
    video_prefix  = os.path.join(prefix, datatype)
    for root_dir, sub_dirs, filenames in os.walk(video_prefix):
        for filename in filenames:
            if filename.endswith('txt'):
                continue
            video_path = os.path.join(root_dir, filename)
            video_paths.append(video_path)

    root = '/6T-4/fengyu/dun'
    logs = './logs'
    if not os.path.exists(root):
        os.makedirs(root)
    l = os.listdir(root)
    flag = 0
    for idx ,video_path in tqdm(enumerate(video_paths)):
        images = []
        cap = cv2.VideoCapture(video_path)
        video_name = video_path.split('/')[-1].split('.')[0].split('_')[0]
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            images.append(frame)
        print("get {} images in {}".format(len(images), video_name))
        img_prefix = os.path.join(root, video_name)
        if not os.path.exists(img_prefix):
            os.makedirs(img_prefix)
        for img_id, image in enumerate(images):
            img_name =video_name + '_' +"%06d"%img_id + '.jpg'
            img_path = os.path.join(img_prefix, img_name)
            cv2.imwrite(img_path, image)
