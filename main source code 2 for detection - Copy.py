import cv2
import numpy as np
from collections import deque

cap = cv2.VideoCapture(0)

cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

# Green color HSV range
lower_color = np.array([20,100,100])
upper_color = np.array([35,255,255])

pts = deque(maxlen=32)

while True:

    ret, frame = cap.read()

    if not ret:
        break

    frame = cv2.flip(frame, 1)

    h, w = frame.shape[:2]

    blurred = cv2.GaussianBlur(frame, (11, 11), 0)

    hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)

    mask = cv2.inRange(hsv, lower_color, upper_color)

    kernel = np.ones((5, 5), np.uint8)

    mask = cv2.erode(mask, kernel, iterations=2)

    mask = cv2.dilate(mask, kernel, iterations=2)

    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    contours, _ = cv2.findContours(
        mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    left_boundary = int(w * 0.35)

    right_boundary = int(w * 0.65)

    top_boundary = int(h * 0.35)

    bottom_boundary = int(h * 0.65)

    if len(contours) > 0:

        c = max(contours, key=cv2.contourArea)

        area = cv2.contourArea(c)

        if area > 800:

            ((x, y), radius) = cv2.minEnclosingCircle(c)

            M = cv2.moments(c)

            if M["m00"] != 0:

                center = (
                    int(M["m10"] / M["m00"]),
                    int(M["m01"] / M["m00"])
                )

                pts.appendleft(center)

                cv2.circle(frame, (int(x), int(y)),
                           int(radius), (0, 255, 255), 2)

                cv2.circle(frame, center,
                           5, (0, 0, 255), -1)

                x_pos = center[0]

                y_pos = center[1]

                if x_pos < left_boundary:

                    direction = "LEFT"

                elif x_pos > right_boundary:

                    direction = "RIGHT"

                elif y_pos < top_boundary:

                    direction = "UP"

                elif y_pos > bottom_boundary:

                    direction = "DOWN"

                else:

                    direction = "CENTER"

                if radius > 120:

                    distance = "TOO CLOSE"

                elif radius < 40:

                    distance = "FAR"

                else:

                    distance = "GOOD"

                text = f"{direction} | {distance}"

                cv2.putText(
                    frame,
                    text,
                    (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0, 255, 0),
                    2
                )

                print(text)

    for i in range(1, len(pts)):

        if pts[i-1] is None or pts[i] is None:

            continue

        thickness = int(np.sqrt(32 / float(i + 1)) * 2)

        cv2.line(
            frame,
            pts[i-1],
            pts[i],
            (255, 0, 0),
            thickness
        )

    cv2.line(frame,
             (left_boundary, 0),
             (left_boundary, h),
             (255, 255, 255), 2)

    cv2.line(frame,
             (right_boundary, 0),
             (right_boundary, h),
             (255, 255, 255), 2)

    cv2.line(frame,
             (0, top_boundary),
             (w, top_boundary),
             (255, 255, 255), 2)

    cv2.line(frame,
             (0, bottom_boundary),
             (w, bottom_boundary),
             (255, 255, 255), 2)

    cv2.imshow("Object Tracking", frame)

    cv2.imshow("Mask", mask)

    if cv2.waitKey(1) & 0xFF == ord('q'):

        break

cap.release()

cv2.destroyAllWindows()
