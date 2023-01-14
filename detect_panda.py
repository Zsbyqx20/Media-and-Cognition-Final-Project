import json
from pathlib import Path

import cv2
import numpy as np
import torch
import yaml

from models.common import DetectMultiBackend
from utils.augmentations import letterbox
from utils.general import Profile, non_max_suppression, scale_boxes
from utils.plots import Annotator, colors


@torch.inference_mode()
def main():
    weights = "data/pretrained/yolov5x.pt"
    source = "data/demo/panda/panda.jpg"
    imgsz = (640, 480)
    classes = [0, 2, 5, 7]
    save_dir = "output/panda"
    data = "data/coco128.yaml"
    split_data = "data/panda-split.yaml"
    refresh_output = True

    save_dir = Path(save_dir)
    json_path = str(save_dir / 'json') + "/panda.json"
    save_path = str(save_dir) + "/panda.jpg"
    print("=> check path existence.")
    if (save_dir / "json").exists() is False:
        (save_dir / "json").mkdir(parents=True)
        print("=> json output path not exist. create a new one already.")

    if refresh_output:
        for path in save_dir.iterdir():
            if path.is_file():
                path.unlink()
            else:
                for subpath in path.iterdir():
                    subpath.unlink()
        print("=> clear history output successfully.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = DetectMultiBackend(weights, device=device, dnn=False, data=data, fp16=False)
    stride, names, pt = model.stride, model.names, model.pt

    model.warmup(imgsz=(1, 3, *imgsz))  # warmup
    seen = 0
    dt = (Profile(), Profile(), Profile())

    with open(split_data) as f:
        info = yaml.safe_load(f.read())
    detect_img = cv2.imread(source)
    
    result_dict = []
    pred_all = []
    for subinfo in info["split_coordinates"]:
        idx = (subinfo["row"] - 1)*4 + subinfo["column"] - 1
        sx = info["original_coordinates"][idx][0]
        sy = info["original_coordinates"][idx][2]
        for subsplit in subinfo["value"]:
            im = detect_img[subsplit[0]+sx:subsplit[1]+sx, subsplit[2]+sy:subsplit[3]+sy]
            im0 = im
            im = letterbox(im, imgsz, stride=stride, auto=pt)[0]
            im = im.transpose((2, 0, 1))[::-1]
            im = np.ascontiguousarray(im)
            with dt[0]:
                im = torch.from_numpy(im).to(model.device)
                im = im.float()
                im /= 255
                im = im[None]
            with dt[1]:
                pred = model(im, augment=False, visualize=False)
            with dt[2]:
                pred = non_max_suppression(pred, conf_thres=0.25, iou_thres=0.45, classes=classes, max_det=1000)
            pred[0][:, :4] = scale_boxes(im.shape[2:], pred[0][:, :4], im0.shape).round()
            pred[0][:,0] += (sy+subsplit[2]); pred[0][:,2] += (sy+subsplit[2]);
            pred[0][:,1] += (sx+subsplit[0]); pred[0][:,3] += (sx+subsplit[0]);
            if pred_all == []:
                pred_all = pred[0]
            else:
                pred_all = torch.cat((pred_all, pred[0]), axis=0)
            print(f"=> finish the calculation of {subsplit}")

    for _, det in enumerate([pred_all]):
        seen += 1
        annotator = Annotator(detect_img, line_width=3, example=str(names))
        if len(det):
            for *xyxy, conf, cls in reversed(det):
                x = int(xyxy[1].item()); y = int(xyxy[0].item()); w = int((xyxy[2] - xyxy[0]).item()); h = int((xyxy[3] - xyxy[1]).item())
                class_label = 2 if int(cls) == 0 else 1
                result = {"category_id":class_label, "bbox":[x,y,w,h]}
                label = f'{names[int(cls)]} {conf:.2f}'
                annotator.box_label(xyxy, label, color=colors(int(cls), True))
                result_dict.append(result)
        detect_img = annotator.result()
        cv2.imwrite(save_path, detect_img)
    result_json = json.dumps(result_dict, indent=4, ensure_ascii=False)
    with open(json_path, 'w') as f:
        f.write(result_json)

if __name__ == '__main__':
    main()