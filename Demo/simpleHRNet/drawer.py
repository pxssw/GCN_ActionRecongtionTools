import cv2
class Drawer(object):
    def __init__(self):
        self.colors = [(255, 0, 0), (255, 85, 0), (255, 170, 0), (255, 255, 0), \
                    (170, 255, 0), (85, 255, 0), (0, 255, 0), (0, 255, 85), (0, 255, 170), \
                    (0, 255, 255), (0, 170, 255), (0, 85, 255), (0, 0, 255), (85, 0, 255), \
                    (170, 0, 255), (255, 0, 255), (255, 0, 170), (255, 0, 85) ]
        self.bones=[
                [13, 12],
                [13, 0 ],
                [ 0, 1 ],
                [ 1, 2 ],
                [13, 3 ],
                [ 3, 4 ],
                [ 4, 5 ],
                [13, 6 ],
                [ 6, 7 ],
                [ 7, 8 ],
                [13, 9 ],
                [ 9, 10],
                [10, 11]
            ]
        self.kpt_names = [
                "r_shoulder", "r_elbow", "r_wrist", 
                "l_shoulder", "l_elbow", "l_wrist",
                "r_hip", "r_knee", "r_ankle",
                "l_hip", "l_knee", "l_ankle",
                "head_top", "upper_neck"
            ]
    def draw_aich_14kp_universal(self, image, kpt):
        assert len(kpt) == 14
        for i,pt in enumerate(kpt):
            if pt[2] > 1.1:
                cv2.circle(image, (int(pt[0]), int(pt[1])), 5, self.colors[i], -1)
            elif pt[2] > 0.2:
                cv2.circle(image, (int(pt[0]), int(pt[1])), 8, (255, 255, 255), -1)

        for i, line in enumerate(self.bones):
            pa = kpt[line[0]]
            pb = kpt[line[1]]
            xa, ya, xb, yb = int(pa[0]),int(pa[1]), int(pb[0]), int(pb[1]) 
            if pa[2]> 0.2 and pb[2] > 0.2:
                cv2.line(image, (xa, ya), (xb, yb), self.colors[i], 5)

            