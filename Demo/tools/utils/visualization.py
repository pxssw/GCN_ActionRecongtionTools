import cv2
import numpy as np
import ipdb

def stgcn_visualize(frame_index,
                    pose,
                    sample_data,
                    edge,
                    video,
                    label=None,
                    confidence=0,
                    height=1080,
                    fps=None):
    _, T, V, M = pose.shape
    T = len(video)
    for t in range(T):
        frame = video[t]
        # image resize
        H, W, c = frame.shape
        # frame = cv2.resize(frame, (height * W // H // 2, height//2))
        frame = cv2.resize(frame, (height * W // H, height))
        H, W, c = frame.shape
        scale_factor = height / 1080
        # draw skeleton
        skeleton = frame * 0
        text = frame * 0
        for m in range(M):
            # ipdb.set_trace()
            # t frame pose doesnt estimated
            if t >= pose.shape[1]:
                # ipdb.set_trace()
                continue
            score = pose[2, t, :, m].max()
            if score < 0.3:
                continue
            for i, j in edge:
                xi = pose[0, t, i, m]
                yi = pose[1, t, i, m]
                xj = pose[0, t, j, m]
                yj = pose[1, t, j, m]
                if xi + yi == 0 or xj + yj == 0:
                    continue
                else:
                    xi = int((xi + 0.5) * W)
                    yi = int((yi + 0.5) * H)
                    xj = int((xj + 0.5) * W)
                    yj = int((yj + 0.5) * H)
                cv2.line(skeleton, (xi, yi), (xj, yj), (255, 255, 255),
                         int(np.ceil(2 * scale_factor)))

        # generate mask
        mask = frame * 0
        # blurred_mask = cv2.blur(mask, (12, 12))

        skeleton_result = mask.astype(float)
        skeleton_result += skeleton.astype(float) * 0.75
        skeleton_result += text.astype(float)
        skeleton_result[skeleton_result > 255] = 255
        skeleton_result.astype(np.uint8)

        rgb_result = mask.astype(float)
        rgb_result += frame.astype(float)
        rgb_result += skeleton.astype(float) * 0.75
        rgb_result[rgb_result > 255] = 255
        rgb_result.astype(np.uint8)


        # sample_results = []
        # H, W, c = frame.shape
        # frame = cv2.resize(frame, (height * W // H // 2, height//2))
        # frame1 = cv2.resize(frame, (W // 5, H))
        # mask = frame1 * 0

        # # image resize
        # _, T, V, M = sample_data.shape
        # for t in range(T):
        #     # draw skeleton
        #     sample_skeleton = frame * 0
        #     # sample_skeleton = frame1 * 0
        #     for m in range(M):
        #         # t frame pose doesnt estimated
        #         score = sample_data[2, t, :, m].max()
        #         if score < 0.3:
        #             continue
        #         for i, j in edge:
        #             xi = sample_data[0, t, i, m]
        #             yi = sample_data[1, t, i, m]
        #             xj = sample_data[0, t, j, m]
        #             yj = sample_data[1, t, j, m]
        #             if xi + yi == 0 or xj + yj == 0:
        #                 continue
        #             else:
        #                 xi = int((xi + 0.5) * W)
        #                 yi = int((yi + 0.5) * H)
        #                 xj = int((xj + 0.5) * W)
        #                 yj = int((yj + 0.5) * H)
        #             cv2.line(sample_skeleton, (xi, yi), (xj, yj), (255, 255, 255),
        #                      5)
        #             # cv2.line(sample_skeleton, (xi, yi//5), (xj, yj//5), (255, 255, 255),
        #             #          5)
        #             # cv2.line(sample_skeleton, (xi, yi), (xj, yj), (255, 255, 255),
        #             #          int(np.ceil(2 * scale_factor)))
        #     sample_result = mask.astype(float)
        #     sample_result += sample_skeleton.astype(float) * 0.75
        #     sample_result.astype(np.uint8)
        #     # ipdb.set_trace()
        #     # cv2.imshow('1', sample_result)
        #     sample_results.append(sample_result)

        put_text(skeleton, 'inputs of MS-G3D', (0.15, 0.5))

        if label is not None:
            label_name = 'voting result: ' + label
            scores = 'Score: ' + str(confidence)
            str_frame_index = 'frame: ' + str(frame_index)


            # label_name = 'voting result: ' + label
            put_text(skeleton_result, label_name, (0.1, 0.5))
            put_text(skeleton_result, scores, (0.3, 0.5))
            put_text(skeleton_result, str_frame_index, (0.5, 0.5))


        if fps is not None:
            put_text(skeleton, 'fps:{:.2f}'.format(fps), (0.9, 0.5))

        img0 = np.concatenate((frame, skeleton), axis=1)
        img1 = np.concatenate((skeleton_result, rgb_result), axis=1)
        # img2 = sample_results[0]
        # for i in range(1, 5):
        #     img2 = np.concatenate((img2,  sample_results[i]), axis=1)
        # img3 = sample_results[5]
        # for i in range(6, 10):
        #     img3 = np.concatenate((img3,  sample_results[i]), axis=1)

        # img2 = sample_results[117]
        # for i in range(118, 122):
        #     img2 = np.concatenate((img2,  sample_results[i]), axis=1)
        # img3 = sample_results[122]
        # for i in range(123, 128):
        #     img3 = np.concatenate((img3,  sample_results[i]), axis=1)
        # ipdb.set_trace()
        # img4 = sample_results[117]
        # for i in range(118, 122):
        #     img4 = np.concatenate((img4,  sample_results[i]), axis=1)
        # img3 = sample_results[122]
        # for i in range(123, 128):
        #     img3 = np.concatenate((img3,  sample_results[i]), axis=1)

        # img3 = sample_results[0]
        # for i in range(1, 2):
        #     img3 = np.concatenate((img3,  sample_results[i]), axis=1)
        # img4 = sample_results[2]
        # for i in range(3, 4):
        #     img4 = np.concatenate((img4,  sample_results[i]), axis=1)

        # img = np.concatenate((img0, img1), axis=0)
        # ipdb.set_trace()
        # img2 = cv2.resize(img2, (img0.shape[1], img0.shape[0]))
        # img3 = cv2.resize(img3, (img0.shape[1], img0.shape[0]))
        # img4 = cv2.resize(img4, (img0.shape[1], img0.shape[0]))

        # img0 = np.concatenate((img0, img3), axis=1)
        # img1 = np.concatenate((img1, img4), axis=1)
        # img = np.concatenate((img0, img1, img2, img3), axis=0)
        # img = np.concatenate((img0, img1, img3), axis=0)
        # ipdb.set_trace()
        # img = np.concatenate((img0, img1, img3, img4, img5, img6), axis=0)
        img = np.concatenate((img0, img1), axis=0)
        yield img


def put_text(img, text, position, scale_factor=1):
    t_w, t_h = cv2.getTextSize(
        text, cv2.FONT_HERSHEY_TRIPLEX, scale_factor, thickness=1)[0]
    H, W, _ = img.shape
    position = (int(W * position[1] - t_w * 0.5),
                int(H * position[0] - t_h * 0.5))
    params = (position, cv2.FONT_HERSHEY_TRIPLEX, scale_factor,
              (255, 255, 255))
    cv2.putText(img, text, *params)


def blend(background, foreground, dx=20, dy=10, fy=0.7):

    foreground = cv2.resize(foreground, (0, 0), fx=fy, fy=fy)
    h, w = foreground.shape[:2]
    b, g, r, a = cv2.split(foreground)
    mask = np.dstack((a, a, a))
    rgb = np.dstack((b, g, r))

    canvas = background[-h-dy:-dy, dx:w+dx]
    imask = mask > 0
    canvas[imask] = rgb[imask]
