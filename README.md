# Media & Cognition Final Project

> This project is made for 2022-2023 autumn semester.
>
> Teacher: Fang Lu
>
> Team: Liu Guohong, Zuo Tianwei, Peng Qinhe, Zhao Han
>
> Before running the code, remember to read `dependencies.txt` first.

## Part I - Stereo Camera

### Task 1.1 Build Camera Pair & Stereo Camera Callibration

For **camera pair building**, `@Zuo` had designed the basic frame structure on PC, and we successfully get it printed out in the lab. Proved by the work afterwards, there isn't any problem with the frame and the structure.

The baseline of the camera pairs is about **6 cm**, and here is the photo of it below:

<p align="middle">
    <img src="./attachments/stereo_camera_frame.jpg" width=400 align=center>
</p>

---

For **stereo camera callibration**, `@Peng` had got the intrinsic and extrinsic matrix with MATLAB Stereo Camera Callibration Toolbox; and for opencv usage, `@Liu` had written some code to change the matrix into what opencv needed, which is saved in the `./data/camera.yml` file.

The matrix K1,D1,K2,D2,R,T,E,F,R1,R2,P1,P2,Q had been saved in the file. To extract specific matrix from `./data/camera.yml`, there is a function `load_stereo_coefficients(path)` taking care of this, located in `utils`, and you can easily find it there.

---

### Task 1.2 Stereo Disparity(Depth) Estimation

For **stereo disparity estimation**, `@Liu` had referred to the work conducted by  `Haofei Xu,etc` in 2022, which is called "**Unifying Flow, Stereo and Depth Estimation**", with one network structure named "unimatch". This is "a unified dense correspondence matching formulation and model for 3 tasks", which include optical flow, disparity and depth estimation. The link to this work is [here](https://arxiv.org/abs/2211.05783 "arxiv"), and the project page is [here](https://haofeixu.github.io/unimatch/ "unimatch").

<p align="middle">
    <img src="./attachments/unimatch_homepage.png" width=400>
</p>

In this project, we have modified the code from that of unimatch, which supports better for opencv frames input. `@Liu` also prepared demo for both image-pair input and video-pair input, since it will be convenient to check the calculation speed for a single frame. (It is disappointing that opencv's `VideoWriter` has a slow speed to write one frame into a video, which is actually about 8~10 times slower than just getting the disparity alone)

You are able to check the demo result of image pairs and video pairs in the `_result` folder. And you can get the same result if you run the code, this time in the `output` folder, and the results are certainly the same. The speed for a single frame or image is about 0.2~0.25 second, the data is got using an GPU (1 TITAN Xp). For video process you will see a much lower result because of low IO speed mentioned above.

<table>
    <tr align="center">
        <td>left image</td>
        <td>right image</td>
        <td>disparity image</td>
    </tr>
    <tr align="center">
        <td><img src="./data/demo/stereo/images/left/left_01.jpg" width="250"/></td>
        <td><img src="./data/demo/stereo/images/right/right_01.jpg" width="250"/></td>
        <td><img src="./data/demo/stereo/images/_result/result_01.png" width="250"/></td>
    </tr>
</table>

As the result above, the **modified model** works fine on the self-captured images. You can tell thin textures and clear boundary of objects, with distinct colors. In the video demo, I was **changing the position** of some object and **adding one small object**. Obviously, the result is smooth and fine.

- For image pairs, run (or configure first, not necessary) `image_stereo.py`;
- For video pairs, run `video_stereo.py`;
- If you are interested in testing your own image/video pair, just put them in `data/demo/stereo/images/left` and `xxx/right` folders; remember to rename your images like `left_<whatever you like>.jpg` and `right_<the same as its left pair.jpg>`. For video pairs of your own, remember to specify both paths in the `.py` file.
- To test on your own pairs, **you should change `./data/camera.yml` first**.

---

## Part II - Object Detection

### Task 2.1 Basic Algorithms of Object Detection

### Task 2.2 Object Detection with Depth

### Task 2.3 PANDA Challenge
