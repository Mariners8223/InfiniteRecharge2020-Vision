import json
import os
import string
from datetime import datetime
import cv2
from processe import get_center, get_image, get_rotation_matrix, newest_save, get_vision_data
import constants
import numpy as np


# mouse callback function
def onClick(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        global hsv, hsvOn, img
        img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        hsv = np.zeros(hsvSize, np.uint8)
        hsvOn = img[y, x]
        cv2.putText(hsv, str(hsvOn), (0, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255))


# Tracker function
def sign(Name, Test, Bigger):
    if Bigger:
        return lambda x: cv2.setTrackbarPos(Test, "Bars", cv2.getTrackbarPos(Name, "Bars")) if x > cv2.getTrackbarPos(
            Test, "Bars") else x
    return lambda x: cv2.setTrackbarPos(Test, "Bars", cv2.getTrackbarPos(Name, "Bars")) if x < cv2.getTrackbarPos(Test,
                                                                                                                  "Bars") else x


# Set Max
def setMax(Max):
    return lambda x: [cv2.setTrackbarPos(f"{Max}H", "Bars", hsvOn[0]), cv2.setTrackbarPos(f"{Max}G", "Bars", hsvOn[1]),
                      cv2.setTrackbarPos(f"{Max}V", "Bars", hsvOn[2])] if x == 1 else x


# saves the setting
def Save(x):
    if x == 1:
        #date = ''.join([i if i in string.digits else "." for i in str(datetime.now())])

        data["min"] = [int(i) for i in data["min"]]
        data["max"] = [int(i) for i in data["max"]]
        data["rotation"] = [float(i) for i in data["rotation"]]
        print(data)
        with open(f'CalibrationOutPuts/default.json', 'w') as outfile:
            json.dump(data, outfile)
        data["rotation"] = np.array(data["rotation"])
        data["min"] = np.array(data["min"])
        data["max"] = np.array(data["max"])


def nothing(x): pass


# declaration of windows and constants
Resize = (constants.WIDTH, constants.HEIGHT)
hsvSize = (60, 640, 3)
cv2.namedWindow("Bars")
cv2.namedWindow("original")
cv2.namedWindow("processed")
cv2.namedWindow("hsv")
# cv2.namedWindow("set")
cv2.namedWindow("save")
cv2.namedWindow("Angle")

# img = cv2.imread('Tester2.jpg')
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 1)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, constants.WIDTH)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, constants.HEIGHT)


hsv = np.zeros(hsvSize, np.uint8)
cv2.setMouseCallback("original", onClick)
hsvOn = np.zeros(3)
data = {}

# Window arrangement
cv2.moveWindow("Bars", 645, 515)
cv2.moveWindow("original", 0, 0)
cv2.moveWindow("processed", 645, 0)
cv2.moveWindow("hsv", 0, 515)
cv2.moveWindow("Angle", 0, 515 + 95)
cv2.moveWindow("save", 322, 515 + 95)  # + 115)

cv2.resizeWindow("Bars", 400, 350)
# cv2.resizeWindow("set", 317, 80)
cv2.resizeWindow("save", 318, 40)
cv2.resizeWindow("Angle", 317, 120)

# Tracker creation
cv2.createTrackbar("MaxH", "Bars", 0, 179, sign("MaxH", "MinH", False))
cv2.createTrackbar("MinH", "Bars", 0, 179, sign("MinH", "MaxH", True))
cv2.createTrackbar("MaxS", "Bars", 0, 255, sign("MaxS", "MinS", False))
cv2.createTrackbar("MinS", "Bars", 0, 255, sign("MinS", "MaxS", True))
cv2.createTrackbar("MaxV", "Bars", 0, 255, sign("MaxV", "MinV", False))
cv2.createTrackbar("MinV", "Bars", 0, 255, sign("MinV", "MaxV", True))
cv2.createTrackbar("AngleX", "Angle", 0, 200, nothing)
cv2.createTrackbar("AngleY", "Angle", 0, 200, nothing)
cv2.createTrackbar("AngleZ", "Angle", 0, 200, nothing)

cv2.createTrackbar("Blur", "Bars", 3, 80, nothing)
cv2.createTrackbar("Light(Neg)", "Bars", 0, 50, lambda x: print("use v4l2-ctl --set-ctrl=exposure_absolute=x"))

# cv2.createTrackbar("SetAsMin", "set", 0, 1, setMax("Min"))
# cv2.createTrackbar("SetAsMax", "set", 0, 1, setMax("Max"))
cv2.createTrackbar("Save?", "save", 0, 1, Save)

# Tracker default
defaultVal = newest_save()
cv2.setTrackbarPos("MaxH", "Bars", defaultVal["max"][0])
cv2.setTrackbarPos("MinH", "Bars", defaultVal["min"][0])
cv2.setTrackbarPos("MaxS", "Bars", defaultVal["max"][1])
cv2.setTrackbarPos("MinS", "Bars", defaultVal["min"][1])
cv2.setTrackbarPos("MaxV", "Bars", defaultVal["max"][2])
cv2.setTrackbarPos("MinV", "Bars", defaultVal["min"][2])

cv2.setTrackbarPos("Light(Neg)", "Bars", -defaultVal["light"])
cv2.setTrackbarPos("Blur", "Bars", defaultVal["blur"])

cv2.setTrackbarPos("AngleX", "Angle", int(defaultVal["rotation"][0] * 100000 + 100))
cv2.setTrackbarPos("AngleY", "Angle", int(defaultVal["rotation"][1] * 100000 + 100))
cv2.setTrackbarPos("AngleZ", "Angle", int((defaultVal["rotation"][2] * 100) / (np.pi / 4) + 100))


def main():
    i = 0
    while 1:
        # get img
        global img
        _, img = cap.read()
        #img = cv2.imread(f"imgs\\Tower{i+147}.png")
        i += 1
        i = (i) % (1000-147)
        # getting information from trackers
        data["min"] = np.array([cv2.getTrackbarPos("MinH", "Bars"), cv2.getTrackbarPos("MinS", "Bars"),
                                cv2.getTrackbarPos("MinV", "Bars")])
        data["max"] = np.array([cv2.getTrackbarPos("MaxH", "Bars"), cv2.getTrackbarPos("MaxS", "Bars"),
                                cv2.getTrackbarPos("MaxV", "Bars")])
        data["blur"] = cv2.getTrackbarPos("Blur", "Bars")
        data["light"] = cv2.getTrackbarPos("Light(Neg)", "Bars")
        data["rotation"] = np.array([(cv2.getTrackbarPos("AngleX", "Angle") - 100) / 100000,
                                     (cv2.getTrackbarPos("AngleY", "Angle") - 100) / 100000,
                                     ((cv2.getTrackbarPos("AngleZ", "Angle") - 100) * (np.pi / 4)) / 100])

        angle = get_rotation_matrix(np.array(data["rotation"]))
        imgR = get_image(img, angle)

        # checking blur
        if data["blur"] % 2 == 0:
            data["blur"] -= 1
            cv2.setTrackbarPos("Blur", "Bars", data["blur"])
        if data["blur"] < 3:
            data["blur"] = 3
            cv2.setTrackbarPos("Blur", "Bars", data["blur"])
        a, d, frame_edited = get_vision_data(imgR, data["min"], data["max"], data["blur"], constants.STICKER_AREA)

        # show the original, edited, hsv and bars
        cv2.imshow("original", imgR)
        cv2.imshow("processed", frame_edited)
        cv2.imshow("hsv", hsv)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break


if __name__ == '__main__':
    main()
