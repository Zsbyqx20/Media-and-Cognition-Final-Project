import warnings
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from rich.progress import track

from models.unimatch import UniMatch
from utils import stereo_util

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

if __name__ == '__main__':
    # you can set the image size and output path of the program here
    # for video pairs you are required to point out where left and right videos are
    inference_size = [480, 640]
    output_path = "./output/stereo/video"

    left_src = "./data/demo/stereo/videos/left/left_video.mp4"
    right_src = "./data/demo/stereo/videos/right/right_video.mp4"

    # the constant below is correspond to the model or result
    # do not edit them unless you know what you are doing
    attn_splits_list = [2,8]
    corr_radius_list = [-1,4]
    prop_radius_list = [-1,2]
    refresh_output = True
    resume = "./data/pretrained/gm_stereo.pth"
    torch.manual_seed(1107)
    torch.cuda.manual_seed(1107)
    np.random.seed(1107)
    torch.backends.cudnn.benchmark = True
    device = torch.device("cuda")
    left_src = Path(left_src)
    right_src = Path(right_src)
    assert left_src.exists() and right_src.exists()
    print("=> initialization start.")
    warnings.filterwarnings('ignore')

    cap_left = cv2.VideoCapture(str(left_src))
    cap_right = cv2.VideoCapture(str(right_src))

    cap_left.set(cv2.CAP_PROP_FRAME_WIDTH, inference_size[1])
    cap_left.set(cv2.CAP_PROP_FRAME_HEIGHT, inference_size[0])
    cap_right.set(cv2.CAP_PROP_FRAME_WIDTH, inference_size[1])
    cap_right.set(cv2.CAP_PROP_FRAME_HEIGHT, inference_size[0])

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    fps = cap_left.get(cv2.CAP_PROP_FPS)
    num_frames = int(cap_left.get(cv2.CAP_PROP_FRAME_COUNT))

    output_path = Path(output_path)
    print("=> check path existence.")
    if output_path.exists() is False:
        output_path.mkdir(parents=True)
        print("=> output path not exist. create a new one already.")

    if refresh_output:
        for path in output_path.iterdir():
            path.unlink()
        print("=> clear history output successfully.")

    save_name = str(output_path / "result.mp4")
    out = cv2.VideoWriter(save_name, fourcc, fps, (inference_size[1], inference_size[0]))

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

    for i in track(range(num_frames), description="[red]Generating..."):
        _, left_frame = cap_left.read()
        _, right_frame = cap_right.read()

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

        out.write(disp)
    
    print(f"=> the result of video pair has been saved into {save_name}.")