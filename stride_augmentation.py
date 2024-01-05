from PIL import Image
import numpy as np
import cv2 as cv
import random
import torch

def random_stride_mask_tensor(tensor):
    # tensor_shape = [1, 3, 256, 256]
    width, height = tensor.shape[2], tensor.shape[3]

    # width, height = img.shape[0], img.shape[1]
    # mask_bg = Image.new('RGB', (width, height))
    mask_bg = np.ones((width, height, 3), np.uint8)
    random_value = random.random()
    print(random_value)
    # vertical
    # 111
    if random_value < 1/12:
    # white-black-white-black
        percent = int(width / 4)

        mask1 = mask_bg
        # [height, width]
        mask1[:, percent:percent*2] = 0
        mask1[:, percent*3:percent*4] = 0

    # 222
    elif 1/12 < random_value < 2/12:
    # white-black-white-black
        percent = int(width / 4)
        mask1 = mask_bg
        # [height, width]
        mask1[:, 0:percent] = 0
        mask1[:, percent*2:percent*3] = 0

    # 333
    elif 2/12 < random_value <3/12:
    # white-black-white-black
        percent = int(width/8)
        mask1 = mask_bg
        # [height, width]
        mask1[:, percent:percent*2] = 0
        mask1[:, percent * 3:percent * 4] = 0
        mask1[:, percent * 5:percent * 6] = 0
        mask1[:, percent * 7:percent * 8] = 0

    # 444
    elif 3/12 < random_value < 4/12:
    # white-black-white-black
        percent = int(width/8)
        mask1 = mask_bg
        # [height, width]
        mask1[:, 0:percent] = 0
        mask1[:, percent*2:percent*3] = 0
        mask1[:, percent*4:percent*5] = 0
        mask1[:, percent*6:percent*7] = 0

    # 555
    elif 4/12 < random_value < 5/12:
    # white-black-white-black
        percent = int(width / 16)
        mask1 = mask_bg
        # [height, width]
        mask1[:, percent:percent * 2] = 0
        mask1[:, percent * 3:percent * 4] = 0
        mask1[:, percent * 5:percent * 6] = 0
        mask1[:, percent * 7:percent * 8] = 0
        mask1[:, percent * 9:percent * 10] = 0
        mask1[:, percent * 11:percent * 12] = 0
        mask1[:, percent * 13:percent * 14] = 0
        mask1[:, percent * 15:percent * 16] = 0

    # 666
    elif 5 / 12 < random_value < 6 / 12:
    # white-black-white-black
        percent = int(width / 16)
        mask1 = mask_bg
        # [height, width]
        mask1[:, 0:percent] = 0
        mask1[:, percent * 2:percent * 3] = 0
        mask1[:, percent * 4:percent * 5] = 0
        mask1[:, percent * 6:percent * 7] = 0
        mask1[:, percent * 8:percent * 9] = 0
        mask1[:, percent * 10:percent * 11] = 0
        mask1[:, percent * 12:percent * 13] = 0
        mask1[:, percent * 14:percent * 15] = 0
        # mask1[:, percent * 6:percent * 7] = 0

    # 777
    # Horizontal
    elif 6/12 < random_value < 7/12:
    # white-black-white-black
        percent = int(width / 4)
        mask1 = mask_bg
        # [height, width]
        mask1[percent:percent * 2, :] = 0
        mask1[percent * 3:percent * 4, :] = 0

    # 888
    elif 7/12 < random_value < 8/12:
    # white-black-white-black
        percent = int(width / 4)
        mask1 = mask_bg
        # [height, width]
        mask1[0:percent, :] = 0
        mask1[percent * 2:percent * 3, :] = 0

    # 999
    elif 8/12 < random_value < 9/12:
    # white-black-white-black
        percent = int(width / 8)
        mask1 = mask_bg
        # [height, width]
        mask1[percent:percent * 2, :] = 0
        mask1[percent * 3:percent * 4, :] = 0
        mask1[percent * 5:percent * 6, :] = 0
        mask1[percent * 7:percent * 8, :] = 0

    # 101010
    elif 9/12 < random_value < 10/12:
        # white-black-white-black
        percent = int(width / 8)
        mask1 = mask_bg
        # [height, width]
        mask1[0:percent, :] = 0
        mask1[percent * 2:percent * 3, :] = 0
        mask1[percent * 4:percent * 5, :] = 0
        mask1[percent * 6:percent * 7, :] = 0

    # 111111
    elif 10/12 < random_value < 11/12:
        # white-black-white-black
        percent = int(width / 16)
        mask1 = mask_bg
        # [height, width]
        mask1[percent:percent * 2, :] = 0
        mask1[percent * 3:percent * 4, :] = 0
        mask1[percent * 5:percent * 6, :] = 0
        mask1[percent * 7:percent * 8, :] = 0
        mask1[percent * 9:percent * 10, :] = 0
        mask1[percent * 11:percent * 12, :] = 0
        mask1[percent * 13:percent * 14, :] = 0
        mask1[percent * 15:percent * 16, :] = 0

    # 121212
    else:
        # white-black-white-black
        percent = int(width / 16)
        mask1 = mask_bg
        # [height, width]
        mask1[0:percent, :] = 0
        mask1[percent * 2:percent * 3, :] = 0
        mask1[percent * 4:percent * 5, :] = 0
        mask1[percent * 6:percent * 7, :] = 0
        mask1[percent * 8:percent * 9, :] = 0
        mask1[percent * 10:percent * 11, :] = 0
        mask1[percent * 12:percent * 13, :] = 0
        mask1[percent * 14:percent * 15, :] = 0
        # mask1[:, percent * 6:percent * 7] = 0

    mask1 = np.reshape(mask1, (3, width, height))
    mask1 = torch.from_numpy(mask1)
    mask1 = torch.unsqueeze(mask1, 0)
    mask_255 = mask1*255

    return mask1


def random_stride_mask(img, ratio_n=None):
    width, height = img.shape[0], img.shape[1]
    # mask_bg = Image.new('RGB', (width, height))
    mask_bg = np.ones((width, height, 3), np.uint8)
    if ratio_n == None:
        random_value = random.random()
    elif ratio_n == 0:
        random_value = 0.08
    elif ratio_n == 1:
        random_value = 0.16
    elif ratio_n == 2:
        random_value = 0.24
    elif ratio_n == 3:
        random_value = 0.32
    elif ratio_n == 4:
        random_value = 0.42
    elif ratio_n == 5:
        random_value = 0.49
    elif ratio_n == 6:
        random_value = 0.58
    elif ratio_n == 7:
        random_value = 0.66
    elif ratio_n == 8:
        random_value = 0.74
    elif ratio_n == 9:
        random_value = 0.83
    elif ratio_n == 10:
        random_value = 0.91
    elif ratio_n == 11:
        random_value = 0.99

    # vertical
    # 111
    if random_value < 1/12:
    # white-black-white-black
        percent = int(width / 4)

        mask1 = mask_bg
        # [height, width]
        mask1[:, percent:percent*2] = 0
        mask1[:, percent*3:percent*4] = 0

    # 222
    elif 1/12 < random_value < 2/12:
    # white-black-white-black
        percent = int(width / 4)
        mask1 = mask_bg
        # [height, width]
        mask1[:, 0:percent] = 0
        mask1[:, percent*2:percent*3] = 0

    # 333
    elif 2/12 < random_value <3/12:
    # white-black-white-black
        percent = int(width/8)
        mask1 = mask_bg
        # [height, width]
        mask1[:, percent:percent*2] = 0
        mask1[:, percent * 3:percent * 4] = 0
        mask1[:, percent * 5:percent * 6] = 0
        mask1[:, percent * 7:percent * 8] = 0

    # 444
    elif 3/12 < random_value < 4/12:
    # white-black-white-black
        percent = int(width/8)
        mask1 = mask_bg
        # [height, width]
        mask1[:, 0:percent] = 0
        mask1[:, percent*2:percent*3] = 0
        mask1[:, percent*4:percent*5] = 0
        mask1[:, percent*6:percent*7] = 0

    # 555
    elif 4/12 < random_value < 5/12:
    # white-black-white-black
        percent = int(width / 16)
        mask1 = mask_bg
        # [height, width]
        mask1[:, percent:percent * 2] = 0
        mask1[:, percent * 3:percent * 4] = 0
        mask1[:, percent * 5:percent * 6] = 0
        mask1[:, percent * 7:percent * 8] = 0
        mask1[:, percent * 9:percent * 10] = 0
        mask1[:, percent * 11:percent * 12] = 0
        mask1[:, percent * 13:percent * 14] = 0
        mask1[:, percent * 15:percent * 16] = 0

    # 666
    elif 5 / 12 < random_value < 6 / 12:
    # white-black-white-black
        percent = int(width / 16)
        mask1 = mask_bg
        # [height, width]
        mask1[:, 0:percent] = 0
        mask1[:, percent * 2:percent * 3] = 0
        mask1[:, percent * 4:percent * 5] = 0
        mask1[:, percent * 6:percent * 7] = 0
        mask1[:, percent * 8:percent * 9] = 0
        mask1[:, percent * 10:percent * 11] = 0
        mask1[:, percent * 12:percent * 13] = 0
        mask1[:, percent * 14:percent * 15] = 0
        # mask1[:, percent * 6:percent * 7] = 0

    # 777
    # Horizontal
    elif 6/12 < random_value < 7/12:
    # white-black-white-black
        percent = int(width / 4)
        mask1 = mask_bg
        # [height, width]
        mask1[percent:percent * 2, :] = 0
        mask1[percent * 3:percent * 4, :] = 0

    # 888
    elif 7/12 < random_value < 8/12:
    # white-black-white-black
        percent = int(width / 4)
        mask1 = mask_bg
        # [height, width]
        mask1[0:percent, :] = 0
        mask1[percent * 2:percent * 3, :] = 0

    # 999
    elif 8/12 < random_value < 9/12:
    # white-black-white-black
        percent = int(width / 8)
        mask1 = mask_bg
        # [height, width]
        mask1[percent:percent * 2, :] = 0
        mask1[percent * 3:percent * 4, :] = 0
        mask1[percent * 5:percent * 6, :] = 0
        mask1[percent * 7:percent * 8, :] = 0

    # 101010
    elif 9/12 < random_value < 10/12:
        # white-black-white-black
        percent = int(width / 8)
        mask1 = mask_bg
        # [height, width]
        mask1[0:percent, :] = 0
        mask1[percent * 2:percent * 3, :] = 0
        mask1[percent * 4:percent * 5, :] = 0
        mask1[percent * 6:percent * 7, :] = 0

    # 111111
    elif 10/12 < random_value < 11/12:
        # white-black-white-black
        percent = int(width / 16)
        mask1 = mask_bg
        # [height, width]
        mask1[percent:percent * 2, :] = 0
        mask1[percent * 3:percent * 4, :] = 0
        mask1[percent * 5:percent * 6, :] = 0
        mask1[percent * 7:percent * 8, :] = 0
        mask1[percent * 9:percent * 10, :] = 0
        mask1[percent * 11:percent * 12, :] = 0
        mask1[percent * 13:percent * 14, :] = 0
        mask1[percent * 15:percent * 16, :] = 0

    # 121212
    else:
        # white-black-white-black
        percent = int(width / 16)
        mask1 = mask_bg
        # [height, width]
        mask1[0:percent, :] = 0
        mask1[percent * 2:percent * 3, :] = 0
        mask1[percent * 4:percent * 5, :] = 0
        mask1[percent * 6:percent * 7, :] = 0
        mask1[percent * 8:percent * 9, :] = 0
        mask1[percent * 10:percent * 11, :] = 0
        mask1[percent * 12:percent * 13, :] = 0
        mask1[percent * 14:percent * 15, :] = 0
        # mask1[:, percent * 6:percent * 7] = 0

    # print("mask1.shape is {}".format(mask1.shape))
    mask_255 = mask1*255

    return mask1



if __name__ == '__main__':
    img_path = "/media/nachuan/TOSHIBA_nachuan/Autoencoder/SDNET2018/P/CP/004-113.jpg"
    # img = Image.open(img_path)
    # print(img.size)
    img = cv.imread(img_path)
    # print(img)
    print(img.shape)


    mask1, mask_255 = random_stride_mask(img)

    img = np.reshape(img, (3, 256, 256))

    # print(mask_255)
    # print(mask_255.shape)

    img_mask = img * mask1



    cv.imshow('img', np.reshape(img, (256, 256, 3)))
    # cv.imshow('flip1', cv.flip(img, 0))
    # cv.imshow('flip2', cv.flip(img, 1))
    # cv.imshow('flip3', cv.flip(img, -1))

    cv.imshow('mask', np.reshape(mask_255, (256, 256, 3)))
    cv.imshow('img_mask', np.reshape(img_mask, (256, 256, 3)))

    cv.waitKey(0)

    # img.show()

