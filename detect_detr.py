import time
import warnings
from pathlib import Path

import torch
import torchvision.transforms as T
from PIL import Image

from utils import detection_utils


@torch.inference_mode()
def main():
    # you can set the input and the output directory of your images here
    image_root = "data/demo/detection/detr/input/"
    output_path = "output/detection/detr/"

    test_depth = False
    depth_dir = "data/demo/detection/depth/input"
    depth_file = "data/demo/detection/depth/input/demo-depth.npy"
    depth_output_path = "output/detection/depth"

    class_range = None
    class_exclude = [67, ]

    # the constant below is correspond to the model or result
    # do not edit them unless you know what you are doing
    refresh_output = True
    warnings.filterwarnings('ignore')
    print("=> initialization start.")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = torch.hub.load('facebookresearch/detr:main', 'detr_resnet50', pretrained=True).to(device)
    model.eval()

    if test_depth:
        image_root = depth_dir
        output_path = depth_output_path
    else:
        depth_file = None

    image_root = Path(image_root)
    output_path = Path(output_path)
    print("=> check path existence.")
    if output_path.exists() is False:
        output_path.mkdir(parents=True)
        print("=> output path not exist. create a new one already.")
    
    if refresh_output:
        for path in output_path.iterdir():
            if path.is_file():
                path.unlink()
            else:
                for subpath in path.iterdir():
                    subpath.unlink()
        print("=> clear history output successfully.")

    transform = T.Compose([
        T.Resize(800),
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    image_list = []
    for path in image_root.iterdir():
        if path.suffix == ".npy" and test_depth:
            continue
        elif path.suffix not in [".png", ".jpg"]:
            print(f"=> file {str(path)} is not a valid format of image; it will be ignored.")
        else:
            image_list.append(path)
    
    for image in image_list:
        save_path = str(output_path / image.stem) + ".jpg"
        im = Image.open(str(image))
        st = time.time()

        img = transform(im).unsqueeze(0).to(device)
        outputs = model(img)
        probas = outputs['pred_logits'].softmax(-1)[0, :, :-1]
        keep = probas.max(-1).values > 0.2

        target_boxes = outputs['pred_boxes'][0, keep].cpu()
        bboxes_scaled = detection_utils.rescale_bboxes(target_boxes, im.size)
        detection_utils.plot_results(im, probas[keep], bboxes_scaled, save_path, class_range=class_range, class_exclude=class_exclude, depth_file=depth_file)
        print(f"=> save result to {save_path}. time cost is {round(time.time() - st, 3)} seconds.")


if __name__ == '__main__':
    main()