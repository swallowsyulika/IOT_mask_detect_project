import cv2
import numpy as np
import tflite_runtime.interpreter as tflite
import urllib3
import time


def requestToThingSpeak():
    # upload value to thingSpeak
    url = "https://api.thingspeak.com/update?api_key="
    key = "your key"
    val = f"&field1={noMaskedNum}"
    r = urllib3.PoolManager().request("GET", url + key + val)
    print(r.status)


# init
MODEL_PATH = "myModel/model.tflite"
FACE_CASCADE_PATH = "myModel/haarcascade_frontalface_default.xml"

# load model
interpreter = tflite.Interpreter(model_path=MODEL_PATH)
interpreter.allocate_tensors()
# set label
label = ['mask', 'face']
# load face roi detector
face_cascade = cv2.CascadeClassifier(FACE_CASCADE_PATH)
# Capture video
cap = cv2.VideoCapture(0)

# first upload value
noMaskedNum = 0
# requestToThingSpeak()
s = time.time()

while True:
    # for each frame
    ret, frame = cap.read()
    # fram to gray color
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # detect face roi
    faces = face_cascade.detectMultiScale(gray, 1.3, 4)

    # reset
    noMaskedNum = 0

    # detect all faces is masked or nomasked
    for (fx, fy, fw, fh) in faces:
        # for each face
        # crop face
        crop = frame[fy:fy+fh, fx:fx+fh]
        crop = cv2.resize(crop, (128, 128))
        crop = np.reshape(crop, [1, 128, 128, 3]) / 255.0
        crop = crop.astype('float32')
        # put crop to model
        interpreter.set_tensor(interpreter.get_input_details()[0]['index'], crop)
        # error msg
        interpreter.invoke()
        # get result of model
        output = interpreter.get_tensor(interpreter.get_output_details()[0]['index'])[0]
        # print ans
        print(output, "  ", np.argmax(output),  "  ", label[np.argmax(output)])
        # count nomasked
        if np.argmax(output) == 1:
            noMaskedNum += 1
        # draw face roi and set text of model's result
        cv2.rectangle(frame, (fx, fy), (fx + fw, fy + fh), (255, 0, 0), 2)
        cv2.putText(frame, label[np.argmax(output)], (fx, fy), cv2.FONT_HERSHEY_SIMPLEX, 1, (200, 255, 255), 2, cv2.LINE_AA)

    # upload value delay 15s (thingSpeak default)
    # !!! IMPORTANT !!!
    # It will take 0.88 ~ 1s time to upload
    # that mean the screen will sleep when it upload.
    # It can be solved by using threading
    # but sometime will encounter the value init
    # so will got incorrect value.
    # In fact, I was lazy to solve it XD
    # !!! IMPORTANT !!!

    # c = time.time()
    # if c - s > 15.5:
    #     requestToThingSpeak()
    #     s = c

    # show frame
    cv2.imshow('img', frame)
    # exit()
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
