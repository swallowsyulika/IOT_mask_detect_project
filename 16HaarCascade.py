import cv2
import numpy as np

face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
mouth_cascade = cv2.CascadeClassifier('haarcascade_smile.xml')

cap = cv2.VideoCapture(0)

num = 0
while True:
    ret, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)     # detectMultiScale(img, scaleFactor, minNeighbors)

    for (fx, fy, fw, fh) in faces:
        cv2.rectangle(frame, (fx, fy), (fx+fw, fy+fh), (255, 0, 0), 2)
        cv2.putText(frame, "face", (fx, fy), cv2.FONT_HERSHEY_SIMPLEX, 1, (200, 255, 255), 2, cv2.LINE_AA)
        roi_gray = gray[fy:fy+fh, fx:fx+fw]
        roi_color = frame[fy:fy+fh, fx:fx+fw]
        mouth = mouth_cascade.detectMultiScale(roi_gray)
        max_size = 0
        mx, my, mw, mh = 0, 0, 0, 0
        for (ex, ey, ew, eh) in mouth:
            if ey >= (fy+fh)/2:
                if (ex+ew)*(ey+eh) > max_size:
                    max_size = (ex+ew)*(ey+eh)
                    mx, my, mw, mh = ex, ey, ew, eh
        cv2.rectangle(roi_color, (mx, my), (mx+mw, my+mh), (0, 255, 0), 2)
        cv2.putText(roi_color, "mouth", (mx, my), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 255, 255), 1, cv2.LINE_AA)
        print(f"frame {num} done!")
        num += 1

    cv2.imshow('img', frame)
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
