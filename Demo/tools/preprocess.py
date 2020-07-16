import sys
sys.path.extend(['../'])
import numpy as np
def pad_2D(data, zaxis=[0, 1], xaxis=[8, 4]):
    N, C, T, V, M = data.shape
    s = np.transpose(data, [0, 4, 2, 3, 1])  # N, C, T, V, M  to  N, M, T, V, C
    # print('pad the null frames with the previous frames')
    # for i_s, skeleton in enumerate(tqdm(s)):  # pad
    for i_s, skeleton in enumerate(s):  # pad
        if skeleton.sum() == 0:
            print(i_s, ' has no skeleton')
        # skeleton (M, T, V, C)
        for i_p, person in enumerate(skeleton):
            # person (T,V,C)
            if person.sum() == 0:
                continue
            if person[0].sum() == 0:
                index = (person.sum(-1).sum(-1) != 0)
                tmp = person[index].copy()
                person *= 0
                person[:len(tmp)] = tmp
            for i_f, frame in enumerate(person):
                if frame.sum() == 0:
                    if person[i_f:].sum() == 0:
                        rest = len(person) - i_f
                        # person not zero concat num tims
                        num = int(np.ceil(rest / i_f))
                        pad = np.concatenate([person[0:i_f] for _ in range(num)], 0)[:rest]
                        s[i_s, i_p, i_f:] = pad
                        break

    # ipdb.set_trace()
    data = np.transpose(s, [0, 4, 2, 3, 1])
    return data

if __name__ == '__main__':
    data = np.load('../data/ntu/xview/val_data.npy')
    pre_normalization(data)
    np.save('../data/ntu/xview/data_val_pre.npy', data)
