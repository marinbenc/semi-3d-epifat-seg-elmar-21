import numpy as np
import matplotlib.pyplot as plt
from skimage.io import imread
import cv2 as cv
from skimage.measure import label
from interpolate_predictions import interpolate

def fill_holes(image):
    '''
    source: https://learnopencv.com/filling-holes-in-an-image-using-opencv-python-c/
    '''
    im_floodfill = image.copy()
    h, w = image.shape[:2]
    mask = np.zeros((h+2, w+2), np.uint8)
    cv.floodFill(im_floodfill, mask, (0,0), 255)
    im_floodfill_inv = cv.bitwise_not(im_floodfill)
    if im_floodfill_inv.sum() > image.sum():
        return image
    else:
        im_out = image | im_floodfill_inv
        return im_out

def keep_largest_component(image):
    '''
    source: https://stackoverflow.com/questions/47540926/get-the-largest-connected-component-of-segmentation-image
    '''
    image_th = image.copy() / image.max()
    labels = label(image_th)
    if labels.max() <= 0:
        return image
    
    largestCC = labels == np.argmax(np.bincount(labels.flat)[1:]) + 1
    return largestCC * image.max()

def remove_noise(image):
    kernel = np.ones((5,5),np.uint8)
    image = cv.morphologyEx(np.float32(image), cv.MORPH_OPEN, kernel)
    return np.uint8(image)

def is_circular(image):
    last_row = image[-1]
    return not np.any(last_row == 0)

def process_image(image):
    original_type = image.dtype
    if not is_circular(image):
        return image * 0
    image = remove_noise(image)
    image = fill_holes(image)
    image = keep_largest_component(image)
    return image.astype(original_type)

if __name__ == "__main__":
    test_image_locations = [
        'datasets/eat/peri_predicted/JFul/043.png',
        'datasets/eat/peri_predicted/JFul/029.png',
        'datasets/eat/peri_predicted/JFul/030.png',
        'datasets/eat/peri_predicted/JFul/036.png',
        'datasets/eat/peri_predicted/JFul/041.png',
        'datasets/eat/peri_predicted/JFul/020.png',
        'datasets/eat/peri_predicted/JFul/028.png',
        'datasets/eat/peri_predicted/JFul/015.png',
        'datasets/eat/peri_predicted/JFul/010.png',
        'datasets/eat/peri_predicted/JFul/018.png',

        # 'datasets/eat/peri_predicted/AXav/043.png',
        # 'datasets/eat/peri_predicted/ACel/029.png',
        # 'datasets/eat/peri_predicted/ACel/030.png',
        # 'datasets/eat/peri_predicted/JMir/036.png',
        # 'datasets/eat/peri_predicted/MPai/047.png',
        # 'datasets/eat/peri_predicted/MPai/041.png',
        # 'datasets/eat/peri_predicted/MSil/020.png',
        # 'datasets/eat/peri_predicted/MSil/028.png',
        # 'datasets/eat/peri_predicted/MSil/029.png',
        # 'datasets/eat/peri_predicted/MSil/030.png',
        # 'datasets/eat/peri_predicted/JMir/036.png',
    ]

    for loc in test_image_locations:
        test_image = imread(loc, as_gray=True)
        plt.imshow(test_image)
        plt.show()
        test_image = process_image(test_image)
        plt.imshow(test_image)
        plt.show()