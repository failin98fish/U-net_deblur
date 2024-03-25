import os
import numpy as np
import cv2

def viz_flow(flow, step=24):
    h, w = flow.shape[:2]
    x, y = np.meshgrid(np.arange(0, w, step), np.arange(0, h, step))
    x = x.flatten()
    y = y.flatten()
    dx = flow[y, x, 0]
    dy = flow[y, x, 1]
    lines = np.vstack([x, y, x + dx, y + dy]).T.reshape(-1, 2, 2)
    lines = np.int32(lines)

    kernel_map = np.zeros((h, w, 3), np.uint8)
    for line in lines:
        pt1 = (line[0, 0], line[0, 1])
        pt2 = (line[1, 0], line[1, 1])
        mag = np.sqrt((pt1[0] - pt2[0]) ** 2 + (pt1[1] - pt2[1]) ** 2)
        color = (0, 255, 0)  # 绿色
        thickness = 1
        cv2.arrowedLine(kernel_map, pt1, pt2, color, thickness, tipLength=0.5, line_type=cv2.LINE_AA)

    return kernel_map

def viz_multiple_flow(multiple_flow, step=8):
    # multiple_flow: (H, W, 2*N)
    H, W, C = multiple_flow.shape
    N = C // 2
    kernel_maps = []
    for i in range(N):
        flow = multiple_flow[:, :, 2 * i:2 * (i + 1)]
        kernel_map = viz_flow(flow, step)
        kernel_maps.append(kernel_map)
    return kernel_maps

for f in os.listdir(os.getcwd()):
    if f.endswith('.npy'):
        name = f.split('.')[0]
        multiple_flow = np.load(f)
        kernel_maps = viz_multiple_flow(multiple_flow)
        for idx, kernel_map in enumerate(kernel_maps):
            cv2.imwrite('{}_kernel_map{}.png'.format(name, idx), kernel_map)