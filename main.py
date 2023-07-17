import cv2
import math
import argparse

def my_custom_face_highlighter(net, frame, conf_thresh=0.7):
    frame_cpy = frame.copy()
    frame_h = frame_cpy.shape[0]
    frame_w = frame_cpy.shape[1]
    blob = cv2.dnn.blobFromImage(frame_cpy, 1.0, (300, 300), [104, 117, 123], True, False)

    net.setInput(blob)
    detections = net.forward()
    face_boxes = []
    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > conf_thresh:
            x1 = int(detections[0, 0, i, 3] * frame_w)
            y1 = int(detections[0, 0, i, 4] * frame_h)
            x2 = int(detections[0, 0, i, 5] * frame_w)
            y2 = int(detections[0, 0, i, 6] * frame_h)
            face_boxes.append([x1, y1, x2, y2])
            cv2.rectangle(frame_cpy, (x1, y1), (x2, y2), (0, 255, 0), int(round(frame_h/150)), 8)
    return frame_cpy, face_boxes


parser = argparse.ArgumentParser()
parser.add_argument('--image')
args = parser.parse_args()

face_proto = "opencv_face_detector.pbtxt"
face_model = "opencv_face_detector_uint8.pb"
age_proto = "age_deploy.prototxt"
age_model = "age_net.caffemodel"
gender_proto = "gender_deploy.prototxt"
gender_model = "gender_net.caffemodel"

model_mean_values = (78.4263377603, 87.7689143744, 114.895847746)
age_categories = ['(0-2)', '(4-6)', '(8-12)', '(15-20)', '(25-32)', '(38-43)', '(48-53)', '(60-100)']
gender_categories = ['Male', 'Female']

face_net = cv2.dnn.readNet(face_model, face_proto)
age_net = cv2.dnn.readNet(age_model, age_proto)
gender_net = cv2.dnn.readNet(gender_model, gender_proto)

video = cv2.VideoCapture(args.image if args.image else 0)
padding = 20
while cv2.waitKey(1) < 0:
    has_frame, frame = video.read()
    if not has_frame:
        cv2.waitKey()
        break

    result_img, face_boxes = my_custom_face_highlighter(face_net, frame)
    if not face_boxes:
        print("No face detected")

    for face_box in face_boxes:
        face = frame[max(0, face_box[1] - padding):
                     min(face_box[3] + padding, frame.shape[0] - 1), max(0, face_box[0] - padding)
                     :min(face_box[2] + padding, frame.shape[1] - 1)]

        blob = cv2.dnn.blobFromImage(face, 1.0, (227, 227), model_mean_values, swapRB=False)
        gender_net.setInput(blob)
        gender_preds = gender_net.forward()
        gender = gender_categories[gender_preds[0].argmax()]
        print(f'Predicted Gender: {gender}')

        age_net.setInput(blob)
        age_preds = age_net.forward()
        age = age_categories[age_preds[0].argmax()]
        print(f'Predicted Age: {age[1:-1]} years')

        cv2.putText(result_img, f'{gender}, {age}', (face_box[0], face_box[1] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2, cv2.LINE_AA)
        cv2.imshow("Detecting age and gender", result_img)
