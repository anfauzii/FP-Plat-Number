import argparse
import os

import cv2

import DetectChars
import DetectPlates
import Preprocess as pp
import imutils

# Module level variables
SCALAR_BLACK = (0.0, 0.0, 0.0)
SCALAR_WHITE = (255.0, 255.0, 255.0)
SCALAR_YELLOW = (0.0, 255.0, 255.0)
SCALAR_GREEN = (0.0, 255.0, 0.0)
SCALAR_RED = (0.0, 0.0, 255.0)
N_VERIFY = 5


def main():
    # argumen untuk image
    ap = argparse.ArgumentParser()
    ap.add_argument("-v", "--video", help="Path to video file")
    ap.add_argument("-i", "--image", help="Path to the image")

    args = vars(ap.parse_args())

    img_original_scene = None
    loop = None
    camera = None

    # cek image
    if args.get("image", True):
        img_original_scene = cv2.imread(args["image"])
        if img_original_scene is None:
            print("Please check again the path of image or argument !")
            loop = False

    # load dan cek KNN Model
    assert DetectChars.loadKNNDataAndTrainKNN(), "KNN can't be loaded !"

    save_number = 0
    prev_license = ""
    licenses_verify = []

    # IMAGE
    if not loop:
        img_original_scene = imutils.resize(img_original_scene, width=720)
        # cv2.imshow("original", img_original_scene)
        imgGrayscale, img_thresh = pp.preprocess(img_original_scene)
        cv2.imshow("threshold", img_thresh)
        img_original_scene = imutils.transform(img_original_scene)
        img_original_scene, new_license = searching(img_original_scene, loop)
        print(f"license plate read from image = {new_license} \n")
        cv2.waitKey(0)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    return

# membuat kotakan disekitar plat
def drawRedRectangleAroundPlate(imgOriginalScene, licPlate):
    p2fRectPoints = cv2.boxPoints(licPlate.rrLocationOfPlateInScene)
    cv2.line(imgOriginalScene, tuple(p2fRectPoints[0]), tuple(p2fRectPoints[1]), SCALAR_RED, 2)
    cv2.line(imgOriginalScene, tuple(p2fRectPoints[1]), tuple(p2fRectPoints[2]), SCALAR_RED, 2)
    cv2.line(imgOriginalScene, tuple(p2fRectPoints[2]), tuple(p2fRectPoints[3]), SCALAR_RED, 2)
    cv2.line(imgOriginalScene, tuple(p2fRectPoints[3]), tuple(p2fRectPoints[0]), SCALAR_RED, 2)

# membuat karakter dalam gambar
def writeLicensePlateCharsOnImage(imgOriginalScene, licPlate):
    ptCenterOfTextAreaX = 0
    ptCenterOfTextAreaY = 0

    ptLowerLeftTextOriginX = 0
    ptLowerLeftTextOriginY = 0

    sceneHeight, sceneWidth, sceneNumChannels = imgOriginalScene.shape
    plateHeight, plateWidth, plateNumChannels = licPlate.imgPlate.shape

    intFontFace = cv2.FONT_HERSHEY_SIMPLEX
    fltFontScale = float(plateHeight) / 30.0
    intFontThickness = int(round(fltFontScale * 1.5))

    textSize, baseline = cv2.getTextSize(licPlate.strChars, intFontFace, fltFontScale,
                                         intFontThickness)

    # Center
    ((intPlateCenterX, intPlateCenterY), (intPlateWidth, intPlateHeight),
     fltCorrectionAngleInDeg) = licPlate.rrLocationOfPlateInScene

    intPlateCenterX = int(intPlateCenterX) 
    intPlateCenterY = int(intPlateCenterY)

    ptCenterOfTextAreaX = int(intPlateCenterX)

    # jika plat diatas 3/4 tinggi gambar
    if intPlateCenterY < (sceneHeight * 0.75):
        ptCenterOfTextAreaY = int(round(intPlateCenterY)) + int(
            round(plateHeight * 1.6))
    # jika plat dibawah 1/4 tinggi gambar
    else:
        ptCenterOfTextAreaY = int(round(intPlateCenterY)) - int(
            round(plateHeight * 1.6))

    textSizeWidth, textSizeHeight = textSize

    ptLowerLeftTextOriginX = int(
        ptCenterOfTextAreaX - (textSizeWidth / 2))
    ptLowerLeftTextOriginY = int(
        ptCenterOfTextAreaY + (textSizeHeight / 2))

    # menulis teks dalam gambar
    cv2.putText(imgOriginalScene, licPlate.strChars, (ptLowerLeftTextOriginX, ptLowerLeftTextOriginY), intFontFace,
                fltFontScale, SCALAR_YELLOW, intFontThickness)


def searching(imgOriginalScene, loop):
    licenses = ""
    if imgOriginalScene is None:
        print("error: image not read from file \n")
        os.system("pause")
        return

    # deteksi plat
    listOfPossiblePlates = DetectPlates.detectPlatesInScene(imgOriginalScene)
    # deteksi karakter dalam plat
    listOfPossiblePlates = DetectChars.detectCharsInPlates(listOfPossiblePlates)

    if not loop:
        cv2.imshow("imgOriginalScene", imgOriginalScene)

    if len(listOfPossiblePlates) == 0:
        if not loop:
            print("no license plates were detected\n")
    else:
        listOfPossiblePlates.sort(key=lambda possiblePlate: len(possiblePlate.strChars), reverse=True)
        licPlate = listOfPossiblePlates[0]
        # tampil plat yg sudah di crop
        if not loop:
            # cv2.imshow("imgPlate", licPlate.imgPlate)  
            cv2.imshow("imgThresh", licPlate.imgThresh)

        if len(licPlate.strChars) == 0:
            if not loop:
                print("no characters were detected\n")
                return

        drawRedRectangleAroundPlate(imgOriginalScene, licPlate)
        writeLicensePlateCharsOnImage(imgOriginalScene, licPlate)
        licenses = licPlate.strChars
        # menampilkan nomor plat yang terbaca
        if not loop:
            print("license plate read from image = " + licPlate.strChars + "\n")

        if not loop:
            cv2.imshow("imgOriginalScene", imgOriginalScene)
            cv2.imwrite("imgOriginalScene.png", imgOriginalScene)

    return imgOriginalScene, licenses

if __name__ == "__main__":
    main()
