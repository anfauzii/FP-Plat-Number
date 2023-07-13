import cv2
import numpy as np

# module level variables
CAL_VAL = np.loadtxt("calibrated_value.txt")
(w, h, rotationx, rotationy, rotationz, panX, panY, stretchX, dist, G_S_F_W, G_S_F_H, A_T_B, A_T_W, T_V, Xtrans,
 Ytrans) = np.loadtxt("calibrated_value.txt")
GAUSSIAN_SMOOTH_FILTER_SIZE = (int(G_S_F_W), int(G_S_F_H))
ADAPTIVE_THRESH_BLOCK_SIZE = int(A_T_B)
ADAPTIVE_THRESH_WEIGHT = int(A_T_W)
THRESHOLD_VALUE = int(T_V)

def preprocess(imgOriginal):

    imgGrayscale = extractValue(imgOriginal)

    imgGrayscale = np.invert(imgGrayscale)
    imgMaxContrastGrayscale = maximizeContrast(imgGrayscale)

    height, width = imgGrayscale.shape

    imgBlurred = np.zeros((height, width, 1), np.uint8)

    imgBlurred = cv2.GaussianBlur(imgMaxContrastGrayscale, GAUSSIAN_SMOOTH_FILTER_SIZE, 0)

    imgThresh = cv2.adaptiveThreshold(imgBlurred, THRESHOLD_VALUE, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                      cv2.THRESH_BINARY_INV, ADAPTIVE_THRESH_BLOCK_SIZE, ADAPTIVE_THRESH_WEIGHT)

    return imgGrayscale, imgThresh

def extractValue(imgOriginal):
    height, width, numChannels = imgOriginal.shape

    imgHSV = np.zeros((height, width, 3), np.uint8)

    imgHSV = cv2.cvtColor(imgOriginal, cv2.COLOR_BGR2HSV)

    imgHue, imgSaturation, imgValue = cv2.split(imgHSV)

    return imgValue

def maximizeContrast(imgGrayscale):
    height, width = imgGrayscale.shape

    imgTopHat = np.zeros((height, width, 1), np.uint8)
    imgBlackHat = np.zeros((height, width, 1), np.uint8)

    structuringElement = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))

    imgTopHat = cv2.morphologyEx(imgGrayscale, cv2.MORPH_TOPHAT, structuringElement)
    imgBlackHat = cv2.morphologyEx(imgGrayscale, cv2.MORPH_BLACKHAT, structuringElement)

    imgGrayscalePlusTopHat = cv2.add(imgGrayscale, imgTopHat)
    imgGrayscalePlusTopHatMinusBlackHat = cv2.subtract(imgGrayscalePlusTopHat, imgBlackHat)

    return imgGrayscalePlusTopHatMinusBlackHat
