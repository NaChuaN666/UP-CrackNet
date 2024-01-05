from multiprocessing import Pool
import cv2 as cv
import os
import time

root_dir = '/model_crack500_results/best/'
out_dir = './outputs_all/'
if not os.path.exists(out_dir):
    os.mkdir(out_dir)


def bilater_Otsu(img_path):
    img = cv.imread(os.path.join(root_dir, img_path))
    img_bilater = cv.bilateralFilter(img, 25, 450, 15)
    print(img_bilater.shape)
    img_bilater = cv.cvtColor(img_bilater, cv.COLOR_BGR2GRAY)
    _, Otsu_map = cv.threshold(img_bilater, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)
    new_path = os.path.join(out_dir, img_path)
    cv.imwrite(new_path, Otsu_map)

def main():
    list_image = os.listdir(root_dir)
    print(list_image)
    workers = os.cpu_count()
    # number of processors used will be equal to workers
    with Pool(workers) as p:
        # p.map(bilater_img, list_image)
        p.map(bilater_Otsu, list_image)

if __name__ == '__main__':
    time1 = time.time()
    main()
    time2 = time.time()
    print("time_cost is {}".format(time2 - time1))
