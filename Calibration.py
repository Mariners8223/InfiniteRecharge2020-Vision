import json
import re
import string
from datetime import datetime
import cv2
from openCVTest import distance_angle_frame
import numpy as np


# mouse callback function
def onClick(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        global rgb, rgbOn
        rgb = np.zeros(rgbSize, np.uint8)
        rgbOn = img[y, x][::-1]
        #hsvOn = cv2.cvtColor(rgbOn, cv2.COLOR_RGB2HSV)
        #print(hsvOn)
        cv2.putText(rgb, str(rgbOn), (0, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255))


# Tracker function
def sign(Name, Test, Bigger):
    if Bigger:
        return lambda x: cv2.setTrackbarPos(Test, "Bars", cv2.getTrackbarPos(Name, "Bars")) if x > cv2.getTrackbarPos(
            Test, "Bars") else x
    return lambda x: cv2.setTrackbarPos(Test, "Bars", cv2.getTrackbarPos(Name, "Bars")) if x < cv2.getTrackbarPos(Test,
                                                                                                                  "Bars") else x


# Set Max
def setMax(Max):
    return lambda x: [cv2.setTrackbarPos(f"{Max}R", "Bars", rgbOn[0]), cv2.setTrackbarPos(f"{Max}G", "Bars", rgbOn[1]),
                      cv2.setTrackbarPos(f"{Max}B", "Bars", rgbOn[2])] if x == 1 else x


# saves the setting
def Save(x):
    if x == 1:
        date = ''.join([i if i in string.digits else "." for i in str(datetime.now())])

        data["min"] = [int(i) for i in data["min"]]
        data["max"] = [int(i) for i in data["max"]]
        print(data)
        with open(f'CalibrationOutPuts\\{date}.json', 'w') as outfile:
            json.dump(data, outfile)
        data["min"] = np.array(data["min"])
        data["max"] = np.array(data["max"])

        '''for i in range(1000):
            _, frame = cap.read()
            cv2.imwrite(f"imgs\\Tower{i}.png", frame)'''


def nothing(x): pass


# declaration of windows and constants
Resize = (640, 480)
rgbSize = (60, 640, 3)
cv2.namedWindow("Bars")
cv2.namedWindow("original")
cv2.namedWindow("processed")
cv2.namedWindow("rgb")
cv2.namedWindow("set")
cv2.namedWindow("save")

# rgb
img = cv2.imread('Tester2.jpg')
cap = cv2.VideoCapture(1)
cap.set(15, -10)

rgb = np.zeros(rgbSize, np.uint8)
cv2.setMouseCallback("original", onClick)
rgbOn = np.zeros(3)
data = {}

# Window arrangement
cv2.moveWindow("Bars", 645, 515)
cv2.moveWindow("original", 0, 0)
cv2.moveWindow("processed", 645, 0)
cv2.moveWindow("rgb", 0, 515)
cv2.moveWindow("set", 0, 515 + 95)
cv2.moveWindow("save", 322, 515 + 95)  # + 115)

cv2.resizeWindow("Bars", 400, 350)
cv2.resizeWindow("set", 317, 80)
cv2.resizeWindow("save", 318, 40)

# Tracker creation
cv2.createTrackbar("MaxR", "Bars", 0, 255, sign("MaxR", "MinR", False))
cv2.createTrackbar("MinR", "Bars", 0, 255, sign("MinR", "MaxR", True))
cv2.createTrackbar("MaxG", "Bars", 0, 255, sign("MaxG", "MinG", False))
cv2.createTrackbar("MinG", "Bars", 0, 255, sign("MinG", "MaxG", True))
cv2.createTrackbar("MaxB", "Bars", 0, 255, sign("MaxB", "MinB", False))
cv2.createTrackbar("MinB", "Bars", 0, 255, sign("MinB", "MaxB", True))

cv2.createTrackbar("Blur", "Bars", 3, 80, nothing)
cv2.createTrackbar("Light(Neg)", "Bars", 0, 50, lambda x: cap.set(15, -x))

cv2.createTrackbar("SetAsMin", "set", 0, 1, setMax("Min"))
cv2.createTrackbar("SetAsMax", "set", 0, 1, setMax("Max"))
cv2.createTrackbar("Save?", "save", 0, 1, Save)

# Tracker default
cv2.setTrackbarPos("MaxR", "Bars", 100)
cv2.setTrackbarPos("MinR", "Bars", 0)
cv2.setTrackbarPos("MaxG", "Bars", 255)
cv2.setTrackbarPos("MinG", "Bars", 180)
cv2.setTrackbarPos("MaxB", "Bars", 255)
cv2.setTrackbarPos("MinB", "Bars", 85)

cv2.setTrackbarPos("Light(Neg)", "Bars", 10)
cv2.setTrackbarPos("Blur", "Bars", 27)


def main():
    i = 1
    while 1:
        # get img
        global img
        #  _, img = cap.read()
        img = cv2.imread(f"imgs\\Tower{i+147}.png")
        i += 1
        i = (i) % (1000-147)
        # getting information from trackers
        data["min"] = np.array([cv2.getTrackbarPos("MinR", "Bars"), cv2.getTrackbarPos("MinG", "Bars"),
                                cv2.getTrackbarPos("MinB", "Bars")])
        data["max"] = np.array([cv2.getTrackbarPos("MaxR", "Bars"), cv2.getTrackbarPos("MaxG", "Bars"),
                                cv2.getTrackbarPos("MaxB", "Bars")])
        data["blur"] = cv2.getTrackbarPos("Blur", "Bars")
        data["light"] = -cv2.getTrackbarPos("Light(Neg)", "Bars")

        # checking blur
        if data["blur"] % 2 == 0:
            data["blur"] -= 1
            cv2.setTrackbarPos("Blur", "Bars", data["blur"])
        if data["blur"] < 3:
            data["blur"] = 3
            cv2.setTrackbarPos("Blur", "Bars", data["blur"])

        D, angle, frame_edited = distance_angle_frame(img, data["min"], data["max"], data["blur"])

        # show the original, edited, rgb and bars
        cv2.imshow("original", img)
        cv2.imshow("processed", frame_edited)
        cv2.imshow("rgb", rgb)

        cv2.waitKey(1)


if __name__ == '__main__':
    main()
