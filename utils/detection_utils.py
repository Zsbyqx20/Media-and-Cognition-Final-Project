import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch

COLORS = [[0.000, 0.447, 0.741], [0.850, 0.325, 0.098], [0.929, 0.694, 0.125],
          [0.494, 0.184, 0.556], [0.466, 0.674, 0.188], [0.301, 0.745, 0.933]]

CLASSES = [
    'N/A', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
    'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'N/A',
    'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse',
    'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'N/A', 'backpack',
    'umbrella', 'N/A', 'N/A', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis',
    'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove',
    'skateboard', 'surfboard', 'tennis racket', 'bottle', 'N/A', 'wine glass',
    'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich',
    'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake',
    'chair', 'couch', 'potted plant', 'bed', 'N/A', 'dining table', 'N/A',
    'N/A', 'toilet', 'N/A', 'tv', 'laptop', 'mouse', 'remote', 'keyboard',
    'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'N/A',
    'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier',
    'toothbrush'
]

def box_cxcywh_to_xyxy(x):
    x_c, y_c, w, h = x.unbind(1)
    b = [(x_c - 0.5 * w), (y_c - 0.5 * h),
         (x_c + 0.5 * w), (y_c + 0.5 * h)]
    return torch.stack(b, dim=1)

def rescale_bboxes(out_bbox, size):
    img_w, img_h = size
    b = box_cxcywh_to_xyxy(out_bbox)
    b = b * torch.tensor([img_w, img_h, img_w, img_h], dtype=torch.float32)
    return b

def plot_results(pil_img, prob, boxes, save_path, class_range:list=None, class_exclude:list=None, depth_file:str=None):
    plt.figure(figsize=(16, 10))
    plt.imshow(pil_img)
    ax = plt.gca()
    colors = COLORS * 100
    for p, (xmin, ymin, xmax, ymax), c in zip(prob, boxes.tolist(), colors):
        cl = p.argmax()
        if cl> len(CLASSES):
            continue
        if (CLASSES[cl] == 'person' and p[cl]<0.9):
            continue
        if class_range is not None and cl not in class_range:
            continue
        if class_exclude is not None and cl in class_exclude:
            continue
        inte_xy = label_choice(torch.Tensor([xmin, ymin, xmax, ymax]), np.asarray(pil_img), depth_file)
        if inte_xy is None:
            continue
        ax.add_patch(plt.Rectangle((xmin, ymin), xmax - xmin, ymax - ymin,
                                   fill=False, color=c, linewidth=3))
        text = f'{CLASSES[cl]}: {p[cl]:0.2f}'
        ax.text(xmin, ymin, text, fontsize=15,
                bbox=dict(facecolor='yellow', alpha=0.5))
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(save_path)
    # plt.show()

def label_choice(candidate:torch.Tensor, original_img:np.ndarray, depth_file:str=None):
    if depth_file is None:
        return candidate
    depth_information = np.load(depth_file)
    undistortion_border = original_img.sum(axis=2)
    depth_information[undistortion_border == 0] = 0

    xstart = int(candidate[1])
    xend = int(candidate[3])
    ystart = int(candidate[0])
    yend = int(candidate[2])

    bbox = depth_information[xstart:xend, ystart:yend]
    sobel_bbox_x = cv2.Sobel(bbox, cv2.CV_64F, 1, 0)
    sobel_bbox_y = cv2.Sobel(bbox, cv2.CV_64F, 0, 1)
    gm = cv2.sqrt(sobel_bbox_x ** 2 + sobel_bbox_y ** 2)
    if gm.max() < 20:
        return None
    else:
        return candidate