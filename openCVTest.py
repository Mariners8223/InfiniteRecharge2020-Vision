import numpy as np
import math
import cv2
import constants
import json


# distance between two points
def d(pt1, pt2):
    return math.sqrt((pt1[0] - pt2[0]) ** 2 + (pt1[1] - pt2[1]) ** 2)


def distance_angle_frame(img, min_color, max_color, blur_val):
    # convert image to hsv

    frame_hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    # threshold
    frame_hsv = cv2.inRange(frame_hsv, min_color, max_color)
    # blur
    frame_hsv = cv2.medianBlur(frame_hsv, blur_val)

    height, width = frame_hsv.shape

    # find objects
    contours, _ = cv2.findContours(frame_hsv, 1, 2)

    # find the object in rectangles and apply formulas
    frame_hsv = cv2.cvtColor(frame_hsv, cv2.COLOR_GRAY2RGB)
    if contours:
        # selects the largest area
        best = [0, 0]
        for i in range(len(contours)):
            # gets the smallest rectangle that block the contour
            rect = cv2.minAreaRect(contours[i])
            # convert to box object
            box = cv2.boxPoints(rect)
            box = np.int0(box)
            # area of the rectangle and save the largest
            area = d(box[0], box[1]) * d(box[0], box[3])
            if area > best[0]:
                best = [area, box]

        box = best[1]
        area = best[0]
        # avoid dividing by 0
        if area == 0:
            return None, None, frame_hsv

        D = constants.FOCAL_LENGTH * math.sqrt(constants.STICKER_AREA / area)
        pixel_middle = (box[0] + box[3]) / 2

        Dx = D * math.sin(constants.FOV * 2 * (pixel_middle[0]) / width)
        Dy = D * math.sin(constants.FOV * 2 * (pixel_middle[1]) / height)
        Dz = D ** 2 - Dx ** 2 - Dy ** 2
        # avoid square root negative number
        if Dz < 0:
            return None, None, frame_hsv
        Dz = math.sqrt(Dz)

        angle = 60 - math.degrees(math.atan(Dz / Dx))

        cv2.drawContours(frame_hsv, [box], -1, (0, 0, 255), 2)

        return D, angle, frame_hsv
    return None, None, frame_hsv


def main():
    # load vision data
    data = json.load(open("CalibrationOutPuts\\default.json", "r"))
    light = data["light"]
    blur = data["blur"]
    min_hsv = np.array(data["min"])
    max_hsv = np.array(data["max"])
    print(min_hsv, max_hsv, 27)
    # camera configuration
    cap = cv2.VideoCapture(1)
    cap.set(15, light)
    i = 0
    while True:
        # reads the frame from the camera
        # _, frame = cap.read()
        frame = cv2.imread(f"imgs\\img{i}.png")
        # get the distance, angle and the edited frame
        D, angle, frame_edited = distance_angle_frame(frame, min_hsv, max_hsv, blur)
        # show the original and edited images
        cv2.imshow("original", frame)
        cv2.imshow("processed", frame_edited)
        print(f"D {D}, angle {angle}")
        if cv2.waitKey(1) & 0xFF == 'q':
            break
        i += 1
        i = i % 1000
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
