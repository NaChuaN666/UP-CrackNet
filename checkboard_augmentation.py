import cv2
import numpy as np
import random
import torch

def generatePattern(CheckerboardSize, Nx_cor, Ny_cor, black_first = True):

    black = np.zeros((CheckerboardSize, CheckerboardSize, 3), np.uint8)
    white = np.zeros((CheckerboardSize, CheckerboardSize, 3), np.uint8)
    black[:] = [0, 0, 0]  # 纯黑色
    white[:] = [1, 1, 1]  # 纯白色


    black_white = np.concatenate([black, white], axis=1)
    black_white2 = black_white
    white_black = np.concatenate([white, black], axis=1)
    white_black2 = white_black

    # 横向连接
    for i in range(1, (Nx_cor + 1) // 2):
        black_white2 = np.concatenate([black_white2, black_white], axis=1)
        white_black2 = np.concatenate([white_black2, white_black], axis=1)

    jj = 0
    black_white3 = black_white2

    if black_first:
        for i in range(0, Ny_cor):
            jj += 1
            if jj % 2 == 1:
                black_white3 = np.concatenate((black_white3, white_black2))
            else:
                black_white3 = np.concatenate((black_white3, black_white2))

    else:
        for i in range(0, Ny_cor):
            jj += 1
            if jj % 2 == 1:
                black_white3 = np.concatenate((white_black2, black_white3))
            else:
                black_white3 = np.concatenate((black_white2, black_white3))

    black_white3_255 = black_white3 * 255
    return black_white3

def random_checkboard_mask(img, ratio_n=None):

    if ratio_n == None:
        random_value = random.random()
    elif ratio_n == 0:
        random_value = 0.15
    elif ratio_n == 1:
        random_value = 0.3
    elif ratio_n == 2:
        random_value = 0.49
    elif ratio_n == 3:
        random_value = 0.65
    elif ratio_n == 4:
        random_value = 0.8
    elif ratio_n == 5:
        random_value = 0.99

    if random_value < 1/6:
        cbs, nx, ny = 128, 1 ,1
        black_first = True
        mask = generatePattern(cbs, nx, ny, black_first)

    elif 1/6 < random_value < 2/6:
        cbs, nx, ny = 128, 1, 1
        black_first = False
        mask = generatePattern(cbs, nx, ny, black_first)

    elif 2/6 < random_value < 3/6:
        cbs, nx, ny = 64, 3, 3
        black_first = True
        mask = generatePattern(cbs, nx, ny, black_first)

    elif 3/6 < random_value < 4/6:
        cbs, nx, ny = 64, 3, 3
        black_first = False
        mask = generatePattern(cbs, nx, ny, black_first)

    elif 4/6 < random_value < 5/6:
        cbs, nx, ny = 32, 7, 7
        black_first = True
        mask = generatePattern(cbs, nx, ny, black_first)

    else:
        cbs, nx, ny = 32, 7, 7
        black_first = False
        mask = generatePattern(cbs, nx, ny, black_first)

    return mask

def random_checkboard_mask_new(img, ratio_n=None):

    if ratio_n == None:
        random_value = torch.rand(1)

    if random_value < 1/6:
        mask = np.load("./ck_mask/ck_0.npy")

    elif 1/6 < random_value < 2/6:
        mask = np.load("./ck_mask/ck_1.npy")

    elif 2/6 < random_value < 3/6:
        mask = np.load("./ck_mask/ck_2.npy")

    elif 3/6 < random_value < 4/6:
        mask = np.load("./ck_mask/ck_3.npy")

    elif 4/6 < random_value < 5/6:
        mask = np.load("./ck_mask/ck_4.npy")

    else:
        mask = np.load("./ck_mask/ck_5.npy")

    return mask

if __name__ == '__main__':
    img_path = "/media/nachuan/TOSHIBA_nachuan/Conditional GAN/0915/crack500_test_ori/a/20160222_114759_1281_721.png"

    img = cv2.imread(img_path)
    print(img.shape)

   


