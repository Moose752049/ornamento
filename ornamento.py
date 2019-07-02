import cv2
import numpy as np
import matplotlib as plt
from os import path
from skimage import feature
import scipy.signal as ss
import sys
import glob

def blacktowhiteRatio(im):
    whitePixelNo=cv2.countNonZero(im)
    blackPixelNo=np.size(im)-whitePixelNo
    ratio=blackPixelNo/whitePixelNo
    return ratio

def cropedImageprocessing(croped,x, y, w, h,im):
    R, G, B = cv2.split(croped)
    ret, bw = cv2.threshold(R, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    bw = cv2.morphologyEx(bw, cv2.MORPH_CLOSE, kernel)
    bw = cv2.morphologyEx(bw, cv2.MORPH_OPEN, kernel)
    # Ithres = cv2.erode(Ithres, kernel, iterations=6)
    bw = cv2.dilate(bw, kernel, iterations=1)
    # Find the largest contour and extract it
    im, contours, hierarchy = cv2.findContours(bw, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    #print(np.size(contours))
    if ((np.size(contours)<300) & (np.size(contours)>70)) | (np.size(contours)<49):
        temp = np.zeros(np.size(contours))

        for i, contour in enumerate(contours):
            contourSize = cv2.contourArea(contour)
            temp[i] = contourSize
        index = (np.argsort(temp))
        [x, y, w, h] = cv2.boundingRect(contours[index[np.size(index) - 1]])
        # cv2.rectangle(Irgb, (x, y), (x+w, y+h), (255, 0, 0), 3)
        cropedImg = croped[y:y + h, x:x + w]  # cropped image
    else:
        cropedImg=croped
    return cropedImg,x, y, w, h,im

#*********main function starts here**************

# Read image
#file_path="F:\\freelancer\\Moose\\images\\images\\20.tif"
dirpath=sys.argv[1]
for img in glob.glob(dirpath):
#for img in glob.glob("F:/freelancer/Moose/images/images/*.tif"):
    print(['processing Image...',img])
    Irgb= cv2.imread(img)
    file_path=img

    R, G, B = cv2.split(Irgb)

    # Do some denosiong on the red chnnale (The red channel gave better result than the gray because it is has more contrast
    Rfilter = cv2.bilateralFilter(R, 25, 25, 10)
    # Threshold image
    ret, Ithres = cv2.threshold(Rfilter, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    Ithres = cv2.morphologyEx(Ithres, cv2.MORPH_CLOSE, kernel)
    Ithres = cv2.morphologyEx(Ithres, cv2.MORPH_OPEN, kernel)
    # Ithres = cv2.erode(Ithres, kernel, iterations=6)
    Ithres = cv2.dilate(Ithres, kernel, iterations=4)
    # Find the largest contour and extract it
    im, contours, hierarchy = cv2.findContours(Ithres, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    maxContour = 0
    temp = np.zeros(np.size(contours))
    for i, contour in enumerate(contours):
        contourSize = cv2.contourArea(contour)
        temp[i] = contourSize
        if contourSize > maxContour:
            maxContour = contourSize
            maxContourData = contour

    index = (np.argsort(temp))  # sorting the countour area
    # print(temp[index[np.size(index)-2]],temp[index[np.size(index)-1]])


    # write the file into same directory
    dir_name = path.dirname(file_path)
    file_name = path.basename(file_path)
    base_name, ext = path.splitext(file_name)
    # drawing two largest countours rectangle

    if (temp[index[np.size(index) - 1]] < 170000) | (
        temp[index[np.size(index) - 1]] > 200000):  # confition for second contour
        [x, y, w, h] = cv2.boundingRect(contours[index[np.size(index) - 1]])
        # cv2.rectangle(Irgb, (x, y), (x+w, y+h), (255, 0, 0), 3)
        croped1 = Irgb[y:y + h, x:x + w]  # cropped image
        croped1, x, y, w, h, im = cropedImageprocessing(Irgb[y:y + h, x:x + w], x, y, w, h, im)  # prcoess again
        # print([x, y, w, h,w/h])
        # print(["balck to white ratio"],blacktowhiteRatio(im))
        blackwhiteRatio = blacktowhiteRatio(im)
        if (blackwhiteRatio < 1.5):
            fn = path.join(dir_name, base_name + "Croped1" + ".png")
            cv2.imwrite(fn, croped1)
        else:
            print('Text Image')

    if (temp[index[np.size(index) - 2]] > 13000):  # confition for second contour
        [x, y, w, h] = cv2.boundingRect(contours[index[np.size(index) - 2]])

        # cv2.rectangle(Irgb, (x, y), (x+w, y+h), (255, 0, 0), 3)
        croped2 = Irgb[y:y + h, x:x + w]
        croped2, x, y, w, h, im = cropedImageprocessing(Irgb[y:y + h, x:x + w], x, y, w, h, im)
        # print([x, y, w, h,w/h])
        # print(["balck to white ratio"], blacktowhiteRatio(im))
        blackwhiteRatio = blacktowhiteRatio(im)
        if (blackwhiteRatio < 1.5):
            fn = path.join(dir_name, base_name + "Croped2" + ".png")
            cv2.imwrite(fn, croped2)
        else:
            print('Text Image')

    '''cv2.namedWindow('asfas',cv2.WINDOW_NORMAL)
    cv2.imshow('asfas',im)'''
    cv2.waitKey(0)


