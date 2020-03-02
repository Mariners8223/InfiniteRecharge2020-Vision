import processe
import cv2
import json
import numpy as np
from networktables import NetworkTables
import constants
import serial


def main():
    # As a client to connect to a robot
    ser = serial.Serial('/dev/ttyS0', 9600, timeout=1)
    # NetworkTables.initialize(server='10.82.23.2')
    # sd = NetworkTables.getTable('SmartDashboard')

    # load vision data
    data = processe.newest_save()
    light = data["light"]
    blur = data["blur"]
    min_hsv = np.array(data["min"])
    max_hsv = np.array(data["max"])
    rotation = processe.get_rotation_matrix(np.array(data["rotation"]))

    # camera configuration
    cap = cv2.VideoCapture(0)
    cap.set(15, light)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, constants.WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, constants.HEIGHT)

    while True:
        _, frame = cap.read()
        frame = processe.get_image(frame, rotation)
        angle, distance, processed = processe.get_vision_data(frame, min_hsv, max_hsv, blur, constants.STICKER_AREA)
        cv2.imshow("before", frame)
        cv2.imshow("after", processed)
        '''if distance is not None:
            velocity = processe.velocity(distance, 0.6)
            if velocity is not None:
                sd.putNumber('vel', float(velocity))
            else:
                sd.putNumber('vel', 0)
        else:
            sd.putNumber('vel', 0)'''
        if angle is not None:
            print(str(angle).encode())
            ser.write(str(angle).encode())
        else:
            ser.write(str(0).encode())
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
