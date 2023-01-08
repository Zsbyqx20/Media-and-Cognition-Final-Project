from __future__ import division

import os
import re
from pathlib import Path

import cv2
import numpy as np
import torch


class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, sample):
        for t in self.transforms:
            sample = t(sample)
        return sample

class ToTensor(object):
    """Convert numpy array to torch tensor"""

    def __init__(self, no_normalize=False):
        self.no_normalize = no_normalize

    def __call__(self, sample):
        left = np.transpose(sample['left'], (2, 0, 1))  # [3, H, W]
        if self.no_normalize:
            sample['left'] = torch.from_numpy(left)
        else:
            sample['left'] = torch.from_numpy(left) / 255.
        right = np.transpose(sample['right'], (2, 0, 1))

        if self.no_normalize:
            sample['right'] = torch.from_numpy(right)
        else:
            sample['right'] = torch.from_numpy(right) / 255.

        # disp = np.expand_dims(sample['disp'], axis=0)  # [1, H, W]
        if 'disp' in sample.keys():
            disp = sample['disp']  # [H, W]
            sample['disp'] = torch.from_numpy(disp)

        return sample

class Normalize(object):
    """Normalize image, with type tensor"""

    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, sample):

        norm_keys = ['left', 'right']

        for key in norm_keys:
            # Images have converted to tensor, with shape [C, H, W]
            for t, m, s in zip(sample[key], self.mean, self.std):
                t.sub_(m).div_(s)

        return sample

def load_stereo_coefficients(path):
    cv_file = cv2.FileStorage(path, cv2.FILE_STORAGE_READ)
    K1 = cv_file.getNode("K1").mat()
    D1 = cv_file.getNode("D1").mat()
    K2 = cv_file.getNode("K2").mat()
    D2 = cv_file.getNode("D2").mat()
    R = cv_file.getNode("R").mat()
    T = cv_file.getNode("T").mat()
    E = cv_file.getNode("E").mat()
    F = cv_file.getNode("F").mat()
    R1 = cv_file.getNode("R1").mat()
    R2 = cv_file.getNode("R2").mat()
    P1 = cv_file.getNode("P1").mat()
    P2 = cv_file.getNode("P2").mat()
    Q = cv_file.getNode("Q").mat()
    cv_file.release()
    return [K1, D1, K2, D2, R, T, E, F, R1, R2, P1, P2, Q]

def parse_image_directory(root:str, left_sub:str="left", right_sub:str="right"):
    left_root = os.path.join(root, left_sub)
    right_root = os.path.join(root, right_sub)
    assert os.path.exists(left_root) and os.path.exists(right_root)

    left_image_list = os.listdir(left_root)
    right_image_list = os.listdir(right_root)

    for idx, file_name in enumerate(left_image_list):
        if file_name.split(".")[-1] not in ["png", "jpg"]:
            print(f"the format of {file_name} cannot be processed! it will be left out.")
            del left_image_list[idx]
    for idx, file_name in enumerate(right_image_list):
        if file_name.split(".")[-1] not in ["png", "jpg"]:
            print(f"the format of {file_name} cannot be processed! it will be left out.")
            del right_image_list[idx]
    result = []
    for left_name in left_image_list:
        prefix = re.findall('\w{3,4}t\_(.*?)\.', left_name)[0]
        for right_name in right_image_list:
            if prefix in right_name:
                pair_left = os.path.join(left_root, left_name)
                pair_right = os.path.join(right_root, right_name)
                result.append({"left": pair_left, "right": pair_right, "prefix": prefix})
                right_image_list.remove(right_name)
    return result


def parse_dir(root:Path, left_sub:str="left", right_sub:str="right"):
    assert root.exists()
    left_img_dir = root / left_sub
    right_img_dir = root / right_sub
    left_img_info = []
    right_img_info = []

    for img in list(left_img_dir.iterdir()):
        metadata = {}
        if img.suffix not in [".jpg", ".png"]:
            print(f"=> file {str(img)} is not a valid format of image; it will be ignored.")
            continue
        metadata["full"] = str(img)
        metadata["prefix"] = img.stem[5:]
        left_img_info.append(metadata)
    for img in list(right_img_dir.iterdir()):
        metadata = {}
        if img.suffix not in [".jpg", ".png"]:
            print(f"=> file {str(img)} is not a valid format of image; it will be ignored.")
            continue
        metadata["full"] = str(img)
        metadata["prefix"] = img.stem[6:]
        right_img_info.append(metadata)
    match_list = []
    for info_l in left_img_info:
        for info_r in right_img_info:
            if info_l["prefix"] == info_r["prefix"]:
                match_list.append({"left": info_l["full"], "right":info_r["full"], "prefix":info_r["prefix"]})
                right_img_info.remove(info_r)
    return match_list


def image_undistortion(left_frame:np.ndarray, right_frame:np.ndarray, coeff_src:str, width:int, height:int):
    K1, D1, K2, D2, _, _, _, _, R1, R2, P1, P2, _ = load_stereo_coefficients(coeff_src)
    leftMapX, leftMapY = cv2.initUndistortRectifyMap(K1, D1, R1, P1, (width, height), cv2.CV_32FC1)
    left_rectified = cv2.remap(left_frame, leftMapX, leftMapY, cv2.INTER_LINEAR, cv2.BORDER_CONSTANT)
    rightMapX, rightMapY = cv2.initUndistortRectifyMap(K2, D2, R2, P2, (width, height), cv2.CV_32FC1)
    right_rectified = cv2.remap(right_frame, rightMapX, rightMapY, cv2.INTER_LINEAR, cv2.BORDER_CONSTANT)

    left_rectified = cv2.cvtColor(left_rectified, cv2.COLOR_BGR2RGB)
    right_rectified = cv2.cvtColor(right_rectified, cv2.COLOR_BGR2RGB)
    return left_rectified, right_rectified