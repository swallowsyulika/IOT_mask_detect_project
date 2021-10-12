import cv2
import numpy as np
import tflite_runtime.interpreter as tflite
import urllib3
from threading import Timer

def requestToThingSpeak():
    if stillRunning:
        Timer(16, requestToThingSpeak).start()
    # upload value to thingSpeak
    url = "https://api.thingspeak.com/update?api_key="
    key = "your key"
    val = f"&field1={noMaskedNum}"
    r = urllib3.PoolManager().request("GET", url + key + val)
    print(r.status, noMaskedNum)


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
stillRunning = True
noMaskedNum = 0
newNum = 0
requestToThingSpeak()

while True:
    # for each frame
    ret, frame = cap.read()
    # fram to gray color
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # detect face roi
    faces = face_cascade.detectMultiScale(gray, 1.3, 4)

    # reset
    newNum = 0

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
            newNum += 1
        # draw face roi and set text of model's result
        cv2.rectangle(frame, (fx, fy), (fx + fw, fy + fh), (255, 0, 0), 2)
        cv2.putText(frame, label[np.argmax(output)], (fx, fy), cv2.FONT_HERSHEY_SIMPLEX, 1, (200, 255, 255), 2, cv2.LINE_AA)

    noMaskedNum = newNum
    # show frame
    cv2.imshow('img', frame)
    # exit()
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
stillRunning = False
