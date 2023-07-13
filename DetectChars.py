import math
import os

import cv2
import numpy as np

import Main
import PossibleChar
import Preprocess

# module level variables
kNearest = cv2.ml.KNearest_create()

MIN_PIXEL_WIDTH = 2
MIN_PIXEL_HEIGHT = 8

MIN_ASPECT_RATIO = 0.25
MAX_ASPECT_RATIO = 1.0

MIN_PIXEL_AREA = 80

MIN_DIAG_SIZE_MULTIPLE_AWAY = 0.01
MAX_DIAG_SIZE_MULTIPLE_AWAY = 8.0

MAX_CHANGE_IN_AREA = 0.5

MAX_CHANGE_IN_WIDTH = 0.8
MAX_CHANGE_IN_HEIGHT = 0.2

MAX_ANGLE_BETWEEN_CHARS = 12.0

MIN_NUMBER_OF_MATCHING_CHARS = 5

RESIZED_CHAR_IMAGE_WIDTH = 20
RESIZED_CHAR_IMAGE_HEIGHT = 30

MIN_CONTOUR_AREA = 100


def loadKNNDataAndTrainKNN():
    allContoursWithData = []  # deklarasi list kosong
    validContoursWithData = []

    try:
        npaClassifications = np.loadtxt("classifications.txt", np.float32)  # read in training classifications
    except:  # if file could not be opened
        print("error, unable to open classifications.txt, exiting program\n")  # show error message
        os.system("pause")
        return False

    try:
        npaFlattenedImages = np.loadtxt("flattened_images.txt", np.float32)  # read in training images
    except:  # if file could not be opened
        print("error, unable to open flattened_images.txt, exiting program\n")  # show error message
        os.system("pause")
        return False

    npaClassifications = npaClassifications.reshape(
        (npaClassifications.size, 1))  # reshape numpy array to 1d, necessary to pass to call to train
    kNearest.setDefaultK(1)
    kNearest.train(npaFlattenedImages, cv2.ml.ROW_SAMPLE, npaClassifications)  # train KNN object
    return True


def detectCharsInPlates(listOfPossiblePlates):
    intPlateCounter = 0
    imgContours = None
    contours = []

    if len(listOfPossiblePlates) == 0:  # if list of possible plates is empty
        return listOfPossiblePlates 

    for possiblePlate in listOfPossiblePlates:  # for each possible plate, this is a big for loop that takes up most of the function

        possiblePlate.imgGrayscale, possiblePlate.imgThresh = Preprocess.preprocess(
            possiblePlate.imgPlate)  # preprocess to get grayscale and threshold images
        # increase size of plate image for easier viewing and char detection
        possiblePlate.imgThresh = cv2.resize(possiblePlate.imgThresh, (0, 0), fx=1.7, fy=1.7)  # 1.6,1.6

        # threshold again to eliminate any gray areas
        thresholdValue, possiblePlate.imgThresh = cv2.threshold(possiblePlate.imgThresh, 0.0, 255.0,
                                                                cv2.THRESH_BINARY | cv2.THRESH_OTSU)

        # find all possible chars in the plate,
        listOfPossibleCharsInPlate = findPossibleCharsInPlate(possiblePlate.imgGrayscale, possiblePlate.imgThresh)

        # given a list of all possible chars, find groups of matching chars within the plate
        listOfListsOfMatchingCharsInPlate = findListOfListsOfMatchingChars(listOfPossibleCharsInPlate)

        if (len(listOfListsOfMatchingCharsInPlate) == 0):  # if no groups of matching chars were found in the plate

            possiblePlate.strChars = ""
            continue

        for i in range(0, len(listOfListsOfMatchingCharsInPlate)):  # within each list of matching chars
            listOfListsOfMatchingCharsInPlate[i].sort(
                key=lambda matchingChar: matchingChar.intCenterX)  # sort chars from left to right
            listOfListsOfMatchingCharsInPlate[i] = removeInnerOverlappingChars(
                listOfListsOfMatchingCharsInPlate[i])  # and remove inner overlapping chars

        # within each possible plate, suppose the longest list of potential matching chars is the actual list of chars
        intLenOfLongestListOfChars = 0
        intIndexOfLongestListOfChars = 0

        # loop through all the vectors of matching chars, get the index of the one with the most chars
        for i in range(0, len(listOfListsOfMatchingCharsInPlate)):
            if len(listOfListsOfMatchingCharsInPlate[i]) > intLenOfLongestListOfChars:
                intLenOfLongestListOfChars = len(listOfListsOfMatchingCharsInPlate[i])
                intIndexOfLongestListOfChars = i

        # suppose that the longest list of matching chars within the plate is the actual list of chars
        longestListOfMatchingCharsInPlate = listOfListsOfMatchingCharsInPlate[intIndexOfLongestListOfChars]

        possiblePlate.strChars = recognizeCharsInPlate(possiblePlate.imgThresh, longestListOfMatchingCharsInPlate)

    # end of big for loop that takes up most of the function

    return listOfPossiblePlates


def findPossibleCharsInPlate(imgGrayscale, imgThresh):
    listOfPossibleChars = []  # this will be the return value
    contours = []
    imgThreshCopy = imgThresh.copy()

    # find all contours in plate
    contours, npaHierarchy = cv2.findContours(imgThreshCopy, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

    for contour in contours:  # for each contour
        possibleChar = PossibleChar.PossibleChar(contour)

        if checkIfPossibleChar(
                possibleChar):  # if contour is a possible char, note this does not compare to other chars (yet) . . .
            listOfPossibleChars.append(possibleChar)  # add to list of possible chars
        # end if
    # end if

    return listOfPossibleChars

# cek kemungkinan karakter
def checkIfPossibleChar(possibleChar):
    if (possibleChar.intBoundingRectArea > MIN_PIXEL_AREA and
            possibleChar.intBoundingRectWidth > MIN_PIXEL_WIDTH and possibleChar.intBoundingRectHeight > MIN_PIXEL_HEIGHT and
            MIN_ASPECT_RATIO < possibleChar.fltAspectRatio < MAX_ASPECT_RATIO):
        return True
    else:
        return False

# mencari list dari list matching karakter
def findListOfListsOfMatchingChars(listOfPossibleChars):
    listOfListsOfMatchingChars = []

    for possibleChar in listOfPossibleChars:
        listOfMatchingChars = findListOfMatchingChars(possibleChar,
                                                      listOfPossibleChars)

        listOfMatchingChars.append(possibleChar)

        if len(
                listOfMatchingChars) < MIN_NUMBER_OF_MATCHING_CHARS:
            continue

        listOfListsOfMatchingChars.append(listOfMatchingChars)

        listOfPossibleCharsWithCurrentMatchesRemoved = []

        listOfPossibleCharsWithCurrentMatchesRemoved = list(set(listOfPossibleChars) - set(listOfMatchingChars))

        recursiveListOfListsOfMatchingChars = findListOfListsOfMatchingChars(
            listOfPossibleCharsWithCurrentMatchesRemoved)

        for recursiveListOfMatchingChars in recursiveListOfListsOfMatchingChars:
            listOfListsOfMatchingChars.append(
                recursiveListOfMatchingChars)

        break 

        print(listOfListsOfMatchingChars)
    return listOfListsOfMatchingChars

# mencari list matching karakter
def findListOfMatchingChars(possibleChar, listOfChars):
    listOfMatchingChars = []

    for possibleMatchingChar in listOfChars:
        if possibleMatchingChar == possibleChar:
            continue

        # compute stuff to see if chars are a match
        fltDistanceBetweenChars = distanceBetweenChars(possibleChar, possibleMatchingChar)

        fltAngleBetweenChars = angleBetweenChars(possibleChar, possibleMatchingChar)

        fltChangeInArea = float(
            abs(possibleMatchingChar.intBoundingRectArea - possibleChar.intBoundingRectArea)) / float(
            possibleChar.intBoundingRectArea)

        fltChangeInWidth = float(
            abs(possibleMatchingChar.intBoundingRectWidth - possibleChar.intBoundingRectWidth)) / float(
            possibleChar.intBoundingRectWidth)
        fltChangeInHeight = float(
            abs(possibleMatchingChar.intBoundingRectHeight - possibleChar.intBoundingRectHeight)) / float(
            possibleChar.intBoundingRectHeight)

        # check if chars match
        if (fltDistanceBetweenChars < (possibleChar.fltDiagonalSize * MAX_DIAG_SIZE_MULTIPLE_AWAY) and
                fltAngleBetweenChars < MAX_ANGLE_BETWEEN_CHARS and
                fltChangeInArea < MAX_CHANGE_IN_AREA and
                fltChangeInWidth < MAX_CHANGE_IN_WIDTH and
                fltChangeInHeight < MAX_CHANGE_IN_HEIGHT):
            listOfMatchingChars.append(
                possibleMatchingChar)  # if the chars are a match, add the current char to list of matching chars

    return listOfMatchingChars


# Pythagorean theorem untuk mengukur jarak antara 2 karakter
def distanceBetweenChars(firstChar, secondChar):
    intX = abs(firstChar.intCenterX - secondChar.intCenterX)
    intY = abs(firstChar.intCenterY - secondChar.intCenterY)

    return math.sqrt((intX ** 2) + (intY ** 2))


# use basic trigonometry (SOH CAH TOA) untuk mengukur angle antara 2 karakter
def angleBetweenChars(firstChar, secondChar):
    fltAdj = float(abs(firstChar.intCenterX - secondChar.intCenterX))
    fltOpp = float(abs(firstChar.intCenterY - secondChar.intCenterY))

    if fltAdj != 0.0:
        fltAngleInRad = math.atan(fltOpp / fltAdj)
    else:
        fltAngleInRad = 1.5708

    fltAngleInDeg = fltAngleInRad * (180.0 / math.pi)  # calculate angle in degrees

    return fltAngleInDeg


# hapus inner overlapping karakter
def removeInnerOverlappingChars(listOfMatchingChars):
    listOfMatchingCharsWithInnerCharRemoved = list(listOfMatchingChars)

    for currentChar in listOfMatchingChars:
        for otherChar in listOfMatchingChars:
            if currentChar != otherChar:
                if distanceBetweenChars(currentChar, otherChar) < (
                        currentChar.fltDiagonalSize * MIN_DIAG_SIZE_MULTIPLE_AWAY):
                    if currentChar.intBoundingRectArea < otherChar.intBoundingRectArea:
                        if currentChar in listOfMatchingCharsWithInnerCharRemoved:
                            listOfMatchingCharsWithInnerCharRemoved.remove(currentChar)
                    else:
                        if otherChar in listOfMatchingCharsWithInnerCharRemoved:
                            listOfMatchingCharsWithInnerCharRemoved.remove(otherChar)

    return listOfMatchingCharsWithInnerCharRemoved


# mengenali karakter dalam plat
def recognizeCharsInPlate(imgThresh, listOfMatchingChars):
    strChars = ""

    height, width = imgThresh.shape

    imgThreshColor = np.zeros((height, width, 3), np.uint8)

    listOfMatchingChars.sort(key=lambda matchingChar: matchingChar.intCenterX)  # sort chars from left to right

    cv2.cvtColor(imgThresh, cv2.COLOR_GRAY2BGR,
                 imgThreshColor)

    for currentChar in listOfMatchingChars:  # for each char in plate
        pt1 = (currentChar.intBoundingRectX, currentChar.intBoundingRectY)
        pt2 = ((currentChar.intBoundingRectX + currentChar.intBoundingRectWidth),
               (currentChar.intBoundingRectY + currentChar.intBoundingRectHeight))

        cv2.rectangle(imgThreshColor, pt1, pt2, Main.SCALAR_GREEN, 2)  # draw green box around the char

        # crop char out of threshold image
        imgROI = imgThresh[
                 currentChar.intBoundingRectY: currentChar.intBoundingRectY + currentChar.intBoundingRectHeight,
                 currentChar.intBoundingRectX: currentChar.intBoundingRectX + currentChar.intBoundingRectWidth]

        imgROIResized = cv2.resize(imgROI, (
        RESIZED_CHAR_IMAGE_WIDTH, RESIZED_CHAR_IMAGE_HEIGHT))  # resize image

        npaROIResized = imgROIResized.reshape(
            (1, RESIZED_CHAR_IMAGE_WIDTH * RESIZED_CHAR_IMAGE_HEIGHT))  # flatten image

        npaROIResized = np.float32(npaROIResized)

        retval, npaResults, neigh_resp, dists = kNearest.findNearest(npaROIResized,
                                                                     k=1)

        strCurrentChar = str(chr(int(npaResults[0][0])))  # get character from results

        strChars = strChars + strCurrentChar  # append current char to full string

    return strChars
