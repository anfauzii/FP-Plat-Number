import argparse
import os
import sys

import cv2
import numpy as np

# module level variables
MIN_CONTOUR_AREA = 100

RESIZED_IMAGE_WIDTH = 20
RESIZED_IMAGE_HEIGHT = 30

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-d", "--image_train",
                    help="path for the images that you're going to train")
    args = vars(ap.parse_args())
    if args.get("image", True):
        imgTrainingNumbers = cv2.imread(args["image_train"])  # training numbers image
        if imgTrainingNumbers is None:
            print
            "error: image not read from file \n\n"
            os.system("pause")
            return
    else:
        print("Please add -d or --image_train argument")

    imgGray = cv2.cvtColor(imgTrainingNumbers, cv2.COLOR_BGR2GRAY)  # get grayscale image
    imgBlurred = cv2.GaussianBlur(imgGray, (5, 5), 0)  # blur

    # filter gambar dari abu-abu menjadi hitam putih
    imgThresh = cv2.adaptiveThreshold(imgBlurred,
                                      255,
                                      cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                      cv2.THRESH_BINARY_INV,
                                      11,
                                      2)

    cv2.imshow("imgThresh", imgThresh)  # tampil threshold image

    imgThreshCopy = imgThresh.copy()  # copy threshimage untuk mencari contour

    imgContours, npaContours, npaHierarchy = cv2.findContours(imgThreshCopy,
                                                              cv2.RETR_EXTERNAL,
                                                              cv2.CHAIN_APPROX_SIMPLE)

    # deklrasi array numpy kosong  untuk file
    npaFlattenedImages = np.empty((0, RESIZED_IMAGE_WIDTH * RESIZED_IMAGE_HEIGHT))

    intClassifications = []  # deklarasi empty classifications list

    intValidChars = [ord('0'), ord('1'), ord('2'), ord('3'), ord('4'), ord('5'), ord('6'), ord('7'), ord('8'), ord('9'),
                     ord('a'), ord('b'), ord('c'), ord('d'), ord('e'), ord('f'), ord('g'), ord('h'), ord('i'), ord('j'),
                     ord('k'), ord('l'), ord('m'), ord('n'), ord('o'), ord('p'), ord('q'), ord('r'), ord('s'), ord('t'),
                     ord('u'), ord('v'), ord('w'), ord('x'), ord('y'), ord('z')]
    # get dan break out bounding rect untuk setiap contour
    for npaContour in npaContours:
        if cv2.contourArea(npaContour) > MIN_CONTOUR_AREA:
            [intX, intY, intW, intH] = cv2.boundingRect(npaContour)  

            # draw rectangle for input
            cv2.rectangle(imgTrainingNumbers,
                          (intX, intY),
                          (intX + intW, intY + intH),
                          (0, 0, 255),  # red
                          2)  # thickness

            imgROI = imgThresh[intY:intY + intH, intX:intX + intW]
            imgROIResized = cv2.resize(imgROI, (RESIZED_IMAGE_WIDTH,
                                                RESIZED_IMAGE_HEIGHT))  # resize image

            cv2.imshow("imgROI", imgROI)  # show cropped out char untuk referensi
            cv2.imshow("imgROIResized", imgROIResized)  # show resized image untuk referensi
            cv2.imshow("training_numbers.png",
                       imgTrainingNumbers)  # show training numbers image

            intChar = cv2.waitKey(0)

            if intChar == 27:  # jika esc key ditekan
                sys.exit()  # exit program
            elif intChar in intValidChars:

                intClassifications.append(
                    intChar)

                npaFlattenedImage = imgROIResized.reshape((1,
                                                           RESIZED_IMAGE_WIDTH * RESIZED_IMAGE_HEIGHT))  # flatten image
                npaFlattenedImages = np.append(npaFlattenedImages, npaFlattenedImage,
                                               0)  # tambah flattened image numpy array saat ini kedalam list

    fltClassifications = np.array(intClassifications,
                                  np.float32)  # convert classifications list of ints to numpy array of floats

    npaClassifications = fltClassifications.reshape(
        (fltClassifications.size, 1))  # flatten numpy array of floats to 1d so we can write to file later

    print
    "\n\ntraining complete !!\n"

    np.savetxt("classifications.txt", npaClassifications)  # write flattened images to file
    np.savetxt("flattened_images.txt", npaFlattenedImages)
    changeCaption() 

    cv2.destroyAllWindows()  # remove windows from memory

    return

def changeCaption():
    data = np.loadtxt("classifications.txt")
    i = 0
    for a in data:
        a = int(round(a))
        if (a == ord('a')):
            data[i] = ord('A')
        if (a == ord('b')):
            data[i] = ord('B')
        if (a == ord('c')):
            data[i] = ord('C')
        if (a == ord('d')):
            data[i] = ord('D')
        if (a == ord('e')):
            data[i] = ord('E')
        if (a == ord('f')):
            data[i] = ord('F')
        if (a == ord('g')):
            data[i] = ord('G')
        if (a == ord('h')):
            data[i] = ord('H')
        if (a == ord('i')):
            data[i] = ord('I')
        if (a == ord('j')):
            data[i] = ord('J')
        if (a == ord('k')):
            data[i] = ord('K')
        if (a == ord('l')):
            data[i] = ord('L')
        if (a == ord('m')):
            data[i] = ord('M')
        if (a == ord('n')):
            data[i] = ord('N')
        if (a == ord('o')):
            data[i] = ord('O')
        if (a == ord('p')):
            data[i] = ord('P')
        if (a == ord('q')):
            data[i] = ord('Q')
        if (a == ord('r')):
            data[i] = ord('R')
        if (a == ord('s')):
            data[i] = ord('S')
        if (a == ord('t')):
            data[i] = ord('T')
        if (a == ord('u')):
            data[i] = ord('U')
        if (a == ord('v')):
            data[i] = ord('V')
        if (a == ord('w')):
            data[i] = ord('W')
        if (a == ord('x')):
            data[i] = ord('X')
        if (a == ord('y')):
            data[i] = ord('Y')
        if (a == ord('z')):
            data[i] = ord('Z')
        i = i + 1

    hasil = np.array(data, np.float32)  # convert classifications list of ints to numpy array of floats
    npaClassifications = hasil.reshape((hasil.size, 1))

    np.savetxt("classifications.txt", npaClassifications)

if __name__ == "__main__":
    main()