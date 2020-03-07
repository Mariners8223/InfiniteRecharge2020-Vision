import os
import numpy as np
import math
import cv2
import constants
import json


# distance between two points
def d(pt1, pt2):
    """ (float, float), (float, float) --> float
    distance between 2 points on scalar coordinate system
    :param pt1: first point
    :param pt2: second point
    :return: distance between 2 points
    """
    return math.sqrt((pt1[0] - pt2[0]) ** 2 + (pt1[1] - pt2[1]) ** 2)


def newest_save():
    # Find the newest calibration output
    # nameList = os.listdir("CalibrationOutPuts")
    Newest = "default.json"
    '''if len(nameList) > 1:
        Newest = max(
            [int(i) for i in [f.replace(".", "")[:-4] for f in nameList if f.endswith('.json')] if i.isdigit()])
        Newest = [i for i in nameList if str(Newest)[-6:] == i[-11:-5]][0]'''
    # load vision data
    return json.load(open(f"CalibrationOutPuts/{Newest}", "r"))


def distance_angle_frame(img, min_color, max_color, blur_val, object_area):
    """ int[][][], int[], int[], int --> float, float
    function that calculates the distance and angle from object by image
    :param img: the raw pixels data
    :param min_color: minimum color to cut
    :param max_color: maximum color to cut
    :param blur_val: blur rate
    :return: distance and angle from object
    """
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
            if abs(area) > best[0]:
                best = [area, box]
        box = best[1]
        area = best[0]
        if area <= 0:
            return None, None, frame_hsv
        cv2.drawContours(frame_hsv, [box], -1, (0, 0, 255), 2)

        D = constants.FOCAL_LENGTH * math.sqrt(object_area / area)
        pixel_middle = (box[0] + box[3]) / 2

        Dx = D * math.sin(constants.FOV * 2 * (pixel_middle[0]) / width)
        Dy = D * math.sin(constants.FOV * 2 * (pixel_middle[1]) / height)
        Dz = D ** 2 - Dx ** 2 - Dy ** 2
        # avoid square root negative number
        if Dz < 0:
            cv2.putText(frame_hsv, f"distance = {None}", (10, 400), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255),
                        1)
            cv2.putText(frame_hsv, f"angle = {None}", (10, 450), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            return None, None, frame_hsv
        Dz = math.sqrt(Dz)

        angle = 60 - math.degrees(math.atan(Dz / Dx))

        cv2.putText(frame_hsv, f"distance = {D}", (10, 430), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.putText(frame_hsv, f"angle = {angle}", (10, 450), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        return D, angle, frame_hsv
    return None, None, frame_hsv


def get_center(img, min_color, max_color, blur_val):
    """ int[][][], int[], int[], int --> float
    function that calculates the ratio from object to middle of image
    :param img: the raw pixels data
    :param min_color: minimum color to cut
    :param max_color: maximum color to cut
    :param blur_val: blur rate
    :return: ratio between object and center
    """
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
            if abs(area) > best[0]:
                best = [area, box]
        box = best[1]
        area = best[0]
        if area == 0:
            return None, frame_hsv
        cv2.drawContours(frame_hsv, [box], -1, (0, 0, 255), 2)

        pixel_middle = (box[0] + box[3]) / 2

        cv2.putText(frame_hsv, f"center = {(pixel_middle[0] / (width / 2)) - 1}", (10, 450),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        return (pixel_middle[0] / (width / 2)) - 1, frame_hsv
    cv2.putText(frame_hsv, f"center = {None}", (10, 450), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    return None, frame_hsv


def get_rotation_matrix(rotation_array):
    """ float[] --> float[][]
    :param rotation_array: angles to fix camera rotation
    :return: rotation matrix
    """
    rotation = np.array([
        [np.cos(rotation_array[2]), -np.sin(rotation_array[2]), 0],
        [np.sin(rotation_array[2]), np.cos(rotation_array[2]), 0],
        [0, 0, 1]]).dot(
        np.array([
            [np.cos(rotation_array[1]), 0, -np.sin(rotation_array[1])],
            [0, 1, 0],
            [-np.sin(rotation_array[1]), 0, np.cos(rotation_array[1])]]))
    rotation = rotation.dot(np.array([
            [1, 0, 0],
            [0, np.cos(rotation_array[0]), -np.sin(rotation_array[0])],
            [0, np.sin(rotation_array[0]), np.cos(rotation_array[0])]]))
    return rotation


def get_image(frame, rotation):
    """ int[][][], float[][] --> int[][][]
    :param frame: raw pixels data
    :param rotation: rotation matrix
    :return: rotated image
    """
    frame = cv2.warpPerspective(frame, rotation, (constants.WIDTH, constants.HEIGHT))
    return frame


def velocity(r, y0):
    if y0 + r > 2.791:
        u1 = np.sqrt(9.81 * r * r / (y0 + r - 2.791))
        u2 = np.sqrt(9.81 * (r + 0.74) * (r + 0.74) / (y0 + r - 1.836))
        l1 = np.sqrt(9.81 * r * r / (y0 + r - 2.209))
        l2 = np.sqrt(9.81 * (r + 0.74) * (r + 0.74) / (y0 + r - 1.684))
        upper = min(u1, u2)
        lower = max(l1, l2)
        return (upper + lower) / 2
    return None


def get_vision_data(img, min_color, max_color, blur_val, object_area):
    # convert image to hsv
    frame_hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    # threshold
    frame_hsv = cv2.inRange(frame_hsv, min_color, max_color)

    frame_hsv = cv2.medianBlur(frame_hsv, blur_val)

    frame_hsv = cv2.dilate(frame_hsv, np.ones((5,5), np.uint8), iterations=1)

    height, width = frame_hsv.shape

    # find objects
    _, contours, _ = cv2.findContours(frame_hsv, 1, 2)

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
            if abs(area) > best[0]:
                best = [area, box]
        box = best[1]
        area = best[0]
        if area == 0:
            return None, None, frame_hsv
        cv2.drawContours(frame_hsv, [box], -1, (0, 0, 255), 2)

        D = constants.FOCAL_LENGTH * math.sqrt(object_area / area)

        pixel_middle = (box[0] + box[2]) / 2

        C = (pixel_middle[0] / (width / 2)) - 1

        cv2.putText(frame_hsv, f"center = {C}", (5, height - 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.putText(frame_hsv, f"distance = {D}", (5, height - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.line(frame_hsv, (int(width/2) - 40, int(height / 2)), (int(width/2) + 40, int(height / 2)), (0, 255, 0), 1)
        cv2.line(frame_hsv, (int(width / 2), int(height/2) - 40), (int(width / 2), int(height/2) + 40), (0, 255, 0), 1)

        return C, D, frame_hsv

    cv2.putText(frame_hsv, f"center = {None}", (5, height - 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    cv2.putText(frame_hsv, f"distance = {None}", (5, height - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    cv2.line(frame_hsv, (int(width / 2) - 40, int(height / 2)), (int(width / 2) + 40, int(height / 2)), (0, 255, 0), 1)
    cv2.line(frame_hsv, (int(width / 2), int(height / 2) - 40), (int(width / 2), int(height / 2) + 40), (0, 255, 0), 1)
    return None, None, frame_hsv


def main():
    # load vision data
    data = newest_save()
    light = data["light"]
    blur = data["blur"]
    min_hsv = np.array(data["min"])
    max_hsv = np.array(data["max"])
    rotation = get_rotation_matrix(np.array(data["rotation"]))
    # camera configuration
    cap = cv2.VideoCapture(0)
    #cap.set(15, light)
    #cap.set(cv2.CAP_PROP_FRAME_WIDTH, constants.WIDTH)
    #cap.set(cv2.CAP_PROP_FRAME_HEIGHT, constants.HEIGHT)
    i = 0
    while True:
        # reads the frame from the camera
        # frame = cv2.imread(f"images/img {i}.png")
        _, frame = cap.read()
        #frame = get_image(frame, rotation)
        cv2.imshow("original", frame)
        # get the distance, angle and the edited frame
        #try:
        #D, angle, frame_edited_D_A = get_vision_data(frame, min_hsv, max_hsv, blur, constants.STICKER_AREA)
        # show the original and edited images
        #cv2.imshow("processed", frame_edited_D_A)
        #except:
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        i += 1
        i = i % 264
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
