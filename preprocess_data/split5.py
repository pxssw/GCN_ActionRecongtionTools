import os
import shutil
from tqdm import tqdm
import ipdb
dun = '/6T-4/fengyu/dun'
video_names = os.listdir(dun)
split_dun1 = '/6T-4/fengyu/dun1/'
spliat_dun2 = '/6T-4/fengyu/dun2/'
spliat_dun3 = '/6T-4/fengyu/dun3/'
spliat_dun4 = '/6T-4/fengyu/dun4/'
spliat_dun5 = '/6T-4/fengyu/dun5/'
length = len(video_names) // 5
l = 0
for i in range(5):
    mv_names = video_names[l:l+length]
    for name in tqdm(mv_names):
        src = os.path.join(dun, name)
        dst = split_dun1
        if not os.path.exists(dst):
            os.mkdir(dst)
        shutil.move(src, dst)
