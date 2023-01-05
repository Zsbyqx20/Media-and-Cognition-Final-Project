import os
import shutil
from pathlib import Path

import cv2
import torch

from models.common import DetectMultiBackend
from utils.dataloaders import LoadImages
from utils.general import Profile, non_max_suppression, scale_boxes, xyxy2xywh
from utils.plots import Annotator, colors


@torch.inference_mode()
def main():
    # modify your weights file and input image directory here
    weights = "data/pretrained/yolov5x.pt"
    imgsz = (640, 480)
    source = "data/demo/detection/yolov5/input"

    data = "data/coco128.yaml"
    classes = None
    save_dir = "output/detection/yolov5"
    refresh_output = True

    # the constant below is correspond to the model or result
    # do not edit them unless you know what you are doing

    if refresh_output:
        shutil.rmtree(save_dir)
        print("=> clear history output successfully.")
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
    if not os.path.exists(os.path.join(save_dir, "labels")):
        os.mkdir(os.path.join(save_dir, "labels"))
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = DetectMultiBackend(weights, device=device, dnn=False, data=data, fp16=False)
    stride, names, pt = model.stride, model.names, model.pt
    dataset = LoadImages(source, img_size=imgsz, stride=stride, auto=pt, vid_stride=1)

    model.warmup(imgsz=(1, 3, *imgsz))  # warmup
    seen = 0
    dt = (Profile(), Profile(), Profile())
    save_dir = Path(save_dir)

    for path, im, im0s, _, s in dataset:
        with dt[0]:
            im = torch.from_numpy(im).to(model.device)
            im = im.float()
            im /= 255
            im = im[None]
        with dt[1]:
            pred = model(im, augment=False, visualize=False)
        with dt[2]:
            pred = non_max_suppression(pred, conf_thres=0.25, iou_thres=0.45, classes=classes, max_det=1000)

        for _, det in enumerate(pred):  # per image
            seen += 1
            p, im0, frame = path, im0s.copy(), getattr(dataset, 'frame', 0)

            p = Path(p)  # to Path
            save_path = str(save_dir / p.name)  # im.jpg
            txt_path = str(save_dir / 'labels' / p.stem) + ('' if dataset.mode == 'image' else f'_{frame}')  # im.txt
            s += '%gx%g ' % im.shape[2:]  # print string
            gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
            annotator = Annotator(im0, line_width=3, example=str(names))
            if len(det):
                det[:, :4] = scale_boxes(im.shape[2:], det[:, :4], im0.shape).round()
                for c in det[:, 5].unique():
                    n = (det[:, 5] == c).sum()  # detections per class
                    s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string
                for *xyxy, conf, cls in reversed(det):
                    xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                    line = (cls, *xywh)  # label format
                    with open(f'{txt_path}.txt', 'a') as f:
                        f.write(('%g ' * len(line)).rstrip() % line + '\n')
                    c = int(cls)  # integer class
                    label = f'{names[c]} {conf:.2f}'
                    annotator.box_label(xyxy, label, color=colors(c, True))
            im0 = annotator.result()
            cv2.imwrite(save_path, im0)
        
        print(f"=> {s}{'' if len(det) else '(no detections), '}{dt[1].dt * 1E3:.1f}ms")

    t = tuple(x.t / seen * 1E3 for x in dt)
    print(f'=> Speed: %.1fms pre-process, %.1fms inference, %.1fms NMS per image at shape {(1, 3, *imgsz)}' % t)
    s = f"\n=> {len(list(save_dir.glob('labels/*.txt')))} labels saved to {save_dir / 'labels'}"
    print(f"=> Results saved to {save_dir}{s}")
    return


if __name__ == '__main__':
    main()
