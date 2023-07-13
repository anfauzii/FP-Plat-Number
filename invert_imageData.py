import argparse
import os

import cv2
import numpy as np

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-d", "--image_train",
                    help="path for the images that you're going to invert")
    args = vars(ap.parse_args())
    if args.get("image", True):
        imgTrainingNumbers = cv2.imread(args["image_train"])  # training numbers image
        if imgTrainingNumbers is None:
            print("error: image not read from file \n\n")
            os.system("pause")
            return
    else:
        print("Please add -d or --image_train argument")

    imgGray = cv2.cvtColor(imgTrainingNumbers, cv2.COLOR_BGR2GRAY)
    imgBlurred = cv2.GaussianBlur(imgGray, (5, 5), 0)

    # filter gambar dari abu-abu menjadi hitam putih
    imgThresh = cv2.adaptiveThreshold(imgBlurred,  # input image
                                      0,
                                      cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                      cv2.THRESH_BINARY_INV,
                                      11,
                                      2)

    imgTrainingNumbers = np.invert(imgTrainingNumbers)
    cv2.imwrite("invert_" + args["image_train"], imgTrainingNumbers)
    cv2.imwrite("imgThresh_" + args["image_train"], imgThresh)
    return

if __name__ == "__main__":
    main()
