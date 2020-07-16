import os
import json
from tqdm import tqdm
import ipdb
def demo():
    output_label_path = '/6T-4/fengyu/output/label'
    root = '/6T-4/fengyu/output/ntu_train'
    root1 = '/6T-4/fengyu/output/ntu_val'
    files = os.listdir(root) + os.listdir(root1)
    train_total_labels = dict()
    val_total_labels = dict()
    for id, f in tqdm(enumerate(files)):
        video_name = f.split('.')[0]
        video = {"has_skeleton": True}
        # ipdb.set_trace()
        # if id == 49453:
        #     ipdb.set_trace()
        video["label_index"] = int(video_name.split('A')[1]) - 1
        video["index"] = id
        tmp = int(video_name.split('C')[1].split('P')[0])
        if tmp == 2 or tmp == 3:
            train_total_labels[video_name] = video
        elif tmp == 1:
            val_total_labels[video_name] = video
    train_output_label_path = os.path.join(output_label_path, "train_label.json")
    val_output_label_path = os.path.join(output_label_path, "val_label.json")
    with open(train_output_label_path, 'w') as outfile:
        json.dump(train_total_labels, outfile, indent = 4)
    with open(val_output_label_path, 'w') as outfile:
        json.dump(val_total_labels, outfile, indent = 4)
    print('generate label complete.')


def main():
    demo()

if __name__ == '__main__':
    main()