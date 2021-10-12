import cv2
import mediapipe as mp

mask_threshold = 0.001

mp_face_detection = mp.solutions.face_detection
mp_drawing = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)
with mp_face_detection.FaceDetection(min_detection_confidence=0.7) as face_detection:
    while cap.isOpened():
        success, image = cap.read()

        image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)

        image.flags.writeable = False
        results = face_detection.process(image)
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        if results.detections:
            for detection in results.detections:
                # mp_drawing.draw_detection(image, detection)
                print(detection)
                image_rows, image_cols, _ = image.shape
                location = detection.location_data

                relative_bounding_box = location.relative_bounding_box

                rect_start_point = mp_drawing._normalized_to_pixel_coordinates(
                    relative_bounding_box.xmin, relative_bounding_box.ymin, image_cols,
                    image_rows)
                rect_end_point = mp_drawing._normalized_to_pixel_coordinates(
                    relative_bounding_box.xmin + relative_bounding_box.width,
                    relative_bounding_box.ymin + +relative_bounding_box.height, image_cols,
                    image_rows)

                print(f"start_p:{rect_start_point}, end_p:{rect_end_point}  ", end="")
                if rect_start_point and rect_end_point:

                    x1 = rect_start_point[0]
                    x2 = rect_end_point[0]
                    y1 = rect_start_point[1]
                    y2 = rect_end_point[1]

                    faceLong = y2 - y1
                    upface = image[y1:y1 + int(1 / 3 * faceLong), x1:x2]
                    downface = image[y1+int(1/3*faceLong):y2, x1:x2]

                    hist1 = cv2.calcHist([upface], [0, 1, 2], None, [256, 256, 256], [0, 256, 0, 256, 0, 256])
                    hist2 = cv2.calcHist([downface], [0, 1, 2], None, [256, 256, 256], [0, 256, 0, 256, 0, 256])
                    cv2.normalize(hist1, hist1, 0, 1.0, cv2.NORM_MINMAX)
                    cv2.normalize(hist2, hist2, 0, 1.0, cv2.NORM_MINMAX)

                    near = cv2.compareHist(hist1, hist2, 0)

                    if near < mask_threshold:
                        print("mask")
                        cv2.rectangle(image, rect_start_point, rect_end_point, (255, 0, 0), 2)
                        cv2.putText(image, "mask", rect_start_point, cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 255, 0), 2)
                    else:
                        print("face")
                        cv2.rectangle(image, rect_start_point, rect_end_point, (255, 0, 0), 2)
                        cv2.putText(image, "face", rect_start_point, cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 255, 0), 2)

        cv2.imshow('MediaPipe Face Detection', image)
        if cv2.waitKey(5) & 0xFF == 27:
            break

cap.release()


