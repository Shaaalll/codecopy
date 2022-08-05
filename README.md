# This is a sample Python script.
from collections import deque
from imutils.video import VideoStream
import cv2
import numpy as np
import time
import argparse
import imutils

# ap = argparse.ArgumentParser()
# ap.add_argument("-b", "--buffer", type=int, default=64, help="max buffer size")
# args = vars(ap.parse_args())

blueLower = (14, 99, 82)
blueUpper = (38, 218, 215)

# pts = deque(maxlen=args["buffer"])


vid = cv2.VideoCapture(0)
while True:

    ret, frame = vid.read()
    #cv2.imshow('frame', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    # if cv2.waitKey(1) & 0xFF == ord('q'):
    #     break
    frame = imutils.resize(frame, width=600)
    blurred = cv2.GaussianBlur(frame, (11, 11), 0)
    hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)

    mask = cv2.inRange(hsv, blueLower, blueUpper)
    mask = cv2.erode(mask, None, iterations=2)
    mask = cv2.dilate(mask, None, iterations=2)
    cv2.imshow("mask", mask)
    cnts = cv2.findContours(mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    center = None

    if len(cnts) > 0:
        c = max(cnts, key=cv2.contourArea)
        ((x, y), radius) = cv2.minEnclosingCircle(c)
        M = cv2.moments(c)
        center = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))
        if radius > 2:
            cv2.circle(frame, center, int(radius), (0, 0, 255), 0)
        #     cv2.circle(frame, center, 5, (0, 0, 255), -1)
        # for i in range(1, len(pts)):
        #     thickness = int(np.sqrt(args["buffer"] / float(i + 1)) * 2.5)
        #     cv2.line(frame, pts[i - 1], pts[i], (0, 0, 255), thickness)

        # compute the rotated bounding box of the contour
        # orig = frame.copy()
        box = cv2.minAreaRect(c)
        box = cv2.cv.BoxPoints(box) if imutils.is_cv2() else cv2.boxPoints(box)
        box = np.array(box, dtype="int")
        height = box[1][1]



        # criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        # objp = np.zeros((6 * 7, 3), np.float32)
        # objp[:, :2] = np.mgrid[0:7, 0:6].T.reshape(-1, 2)
        #
        # for fname in frame:
        #     img = cv2.imread(fname)
        #     gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        #     ret, corners = cv.findChessboardCorners(gray, (7, 6), None)
        #     if ret == True:
        #         objpoints.append(objp)



        # order the points in the contour such that they appear
        # in top-left, top-right, bottom-right, and bottom-left
        # order, then draw the outline of the rotated bounding
        # box
        # box = perspective.order_points(box)
        cv2.drawContours(frame, [box.astype("int")], -1, (0, 255, 0), 2)
        print(box)

        # (tl, tr, br, bl) = box
        # (tltrX, tltrY) = midpoint(tl, tr)
        # (blbrX, blbrY) = midpoint(bl, br)
        #
        # # compute the midpoint between the top-left and top-right points,
        # # followed by the midpoint between the top-righ and bottom-right
        # (tlblX, tlblY) = midpoint(tl, bl)
        # (trbrX, trbrY) = midpoint(tr, br)



    cv2.imshow('frame', frame)









vid.release()
cv2.destroyAllWindows()



# vs = VideoStream(src=0).start()
# time.sleep(2.0)
#
# while True:
#     frame = vs.read()
#     frame = frame[1] if args.get()

# Press ⌃R to execute it or replace it with your code.
# Press Double ⇧ to search everywhere for classes, files, tool windows, actions, and settings.
