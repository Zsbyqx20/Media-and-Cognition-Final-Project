from pathlib import Path

import cv2
import numpy as np

pic = cv2.imread("data/demo/panda/panda.jpg")

h_num = 4
w_num = 4

(width, height, depth) = pic.shape

w_cut = int(width / w_num)
h_cut = int(height / h_num)

result = np.zeros((w_cut, h_cut, depth))
position_matrix = np.load("pm.npy").astype(np.uint16)
adjust_matrix = np.zeros((h_num * w_num, 4), dtype=np.uint16)
cnt = 0
adjust_matrix[3] = [2000, 0, 0, 0]          # 1-4
adjust_matrix[4] = [500, 800, 0, 0]         # 2-1
adjust_matrix[5] = [0, -800, 0, -300]       # 2-2
adjust_matrix[6] = [-300, 0, -300, -950]       # 2-3
adjust_matrix[7] = [-150, 500, -950, -400]           # 2-4
adjust_matrix[9] = [-1000, 0, 0, -300]      # 3-2
adjust_matrix[10] = [-300, 0, -300, 0]         # 3-3

for subpath in Path("data/demo/panda/split").iterdir():
    if subpath.is_file():
        subpath.unlink()
print("=> clear history output.")

position_matrix += adjust_matrix
for i in range(w_num):
    for j in range(h_num):
        # si = i*w_cut; ei = (i+1)*w_cut if i < h_num - 1 else width
        # sj = j*h_cut; ej = (j+1)*h_cut if j < w_num - 1 else height
        (si, ei, sj, ej) = position_matrix[cnt]
        result = pic[si:ei, sj:ej, :]
        path = "data/demo/panda/split/" + f'{i+1}_{j+1}.jpg'
        # position_matrix[cnt] = [si, ei, sj, ej]
        cv2.imwrite(path, result)
        print(f"=> split {i+1}_{j+1} has been saved; position for {si, ei, sj, ej}.")
        cnt += 1

# print(position_matrix)
np.save("divide.npy", position_matrix)
# 2-1, 2-2, 3-2, (2-3, 3-3), (1-4, 2-4), 4-1, 4-2
# 1-1, 1-2, 3-1, 1-3, 3-4, 4-4

# finished
# 4-1, 4-2, 2-1, 2-2, 3-2, 2-4

# TODO
# 1-4, 2-3, 3-3