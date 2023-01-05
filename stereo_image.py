import os
import time
import warnings

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from models.unimatch import UniMatch
from utils import stereo_util

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


if __name__ == '__main__':
    # you can set the image size and output path of the program here
    # for image pairs the program will automatically examine available pairs, just set the root path
    inference_size = [480, 640]
    image_pair_root = "./data/demo/stereo/images"
    output_path = "./output/stereo/image"

    # the constant below is correspond to the model or result
    # do not edit them unless you know what you are doing
    attn_splits_list = [2,8]
    corr_radius_list = [-1,4]
    prop_radius_list = [-1,2]
    refresh_output = True
    resume = "./data/pretrained/gmstereo.pth"
    torch.manual_seed(1107)
    torch.cuda.manual_seed(1107)
    np.random.seed(1107)
    torch.backends.cudnn.benchmark = True
    device = torch.device("cuda")
    warnings.filterwarnings('ignore')

    if refresh_output:
        for history_output in os.listdir(output_path):
            os.remove(os.path.join(output_path, history_output))
        print("=> clear history output successfully.")

    model = UniMatch(feature_channels=128, num_scales=2, upsample_factor=4, 
        num_head=1, ffn_dim_expansion=4, num_transformer_layers=6,
        reg_refine=True, task="stereo"
    ).to(device)
    checkpoint = torch.load(resume, map_location="cuda:0")
    model.load_state_dict(checkpoint['model'], strict=False)
    model.eval()
    val_transform_list = [
        stereo_util.ToTensor(),
        stereo_util.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
    ]
    val_transform = stereo_util.Compose(val_transform_list)

    image_pairs = stereo_util.parse_image_directory(image_pair_root)
    print(f"=> {len(image_pairs)} pairs of images has been found.")
    for idx, image_pair in enumerate(image_pairs):
        print(f"=> start processing pair {idx + 1}.")
        start_time = time.time()
        left_frame = np.array(Image.open(image_pair["left"]).convert("RGB")).astype(np.float32)
        right_frame = np.array(Image.open(image_pair["right"]).convert("RGB")).astype(np.float32)

        left_rect, right_rect = stereo_util.image_undistortion(left_frame, right_frame, "./data/camera.yml", inference_size[1], inference_size[0])
        sample = {"left":left_rect, "right":right_rect}
        sample = val_transform(sample)
        left:torch.Tensor = sample["left"].to(device).unsqueeze(0)
        right:torch.Tensor = sample["right"].to(device).unsqueeze(0)

        ori_size = left.shape[-2:]
        if inference_size[0] != ori_size[0] or inference_size[1] != ori_size[1]:
            left = F.interpolate(left, size=inference_size,mode='bilinear', align_corners=True)
            right = F.interpolate(right, size=inference_size,mode='bilinear', align_corners=True)
        
        with torch.no_grad():
            pred_disp = model(
                left, right,
                attn_type="self_swin2d_cross_swin1d",
                attn_splits_list=attn_splits_list,
                corr_radius_list=corr_radius_list,
                prop_radius_list=prop_radius_list,
                num_reg_refine=3,
                task="stereo"
            )["flow_preds"][-1]
        
        if inference_size[0] != ori_size[0] or inference_size[1] != ori_size[1]:
            pred_disp:torch.Tensor = F.interpolate(
                pred_disp.unsqueeze(1), size=ori_size,
                mode='bilinear', align_corners=True
            ).squeeze(1)
            pred_disp = pred_disp * ori_size[-1] / float(inference_size[-1])
        
        disp:np.ndarray = pred_disp[0].cpu().numpy()
        disp = (disp - disp.min()) / (disp.max() - disp.min()) * 255.0
        disp = disp.astype("uint8")
        disp_return = disp
        disp = cv2.applyColorMap(disp, cv2.COLORMAP_INFERNO)

        save_name = "result_" + image_pair["prefix"] + ".png"
        save_name = os.path.join(output_path, save_name)

        cv2.imwrite(save_name, disp)
        print(f"=> the result of pair {idx + 1} has been saved into {save_name}.")
        print(f"=> time cost for pair {idx + 1} is {round(time.time() - start_time, 3)} seconds.")