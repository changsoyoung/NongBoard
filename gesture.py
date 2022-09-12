import cv2
import argparse
import numpy as np

from utils_display import DisplayHand
from utils_mediapipe import MediaPipeHand
from utils_joint_angle import GestureRecognition


parser = argparse.ArgumentParser() 
parser.add_argument('-m', '--mode', default='eval', help='train / eval')
args = parser.parse_args()
mode = args.mode

# mediapipe hand class에서 불러오기
pipe = MediaPipeHand(static_image_mode=False, max_num_hands=2)

# display class에서 불러오기
disp = DisplayHand(max_num_hands=2)

# 웹캠 사용하여 캡쳐하기
cap = cv2.VideoCapture(0)

# gesture recognition class에서 불러오기
gest = GestureRecognition(mode)

counter = 0
class_label = 0
while cap.isOpened():
    ret, img = cap.read()
    if not ret:
        break

    # 거울모드로 바꾸기
    img = cv2.flip(img, 1)

    img.flags.writeable = False

    param = pipe.forward(img)
    zeroarr = [0 for i in range(15)]

    if (param[0]['class'] is not None) and (param[1]['class'] is None) and (mode=='eval'):
        if (param[0]['class'] == 'Left') and (param[1]['class'] is None):
            b = np.concatenate((param[0]['angle'], np.array(zeroarr)))
            print(b)
            param[0]['gesture'] = gest.eval(b)
        if (param[0]['class'] == 'Right') and (param[1]['class'] is None):
            b = np.concatenate((np.array(zeroarr), param[0]['angle']))
            print(b)
            param[0]['gesture'] = gest.eval(b)
    elif (param[0]['class'] is not None) and (param[1]['class'] is not None) and (mode=='eval'):
        if (param[0]['class'] == 'Left'):
            b = np.concatenate((param[0]['angle'], param[1]['angle']))
            print(b)
            param[0]['gesture'] = gest.eval(b)
        if (param[0]['class'] == 'Right'):
            b = np.concatenate((param[1]['angle'], param[0]['angle']))
            print(b)
            param[0]['gesture'] = gest.eval(b)

    img.flags.writeable = True

    # 키보드 입력부분
    cv2.imshow('img 2D', disp.draw2d(img.copy(), param))

    key = cv2.waitKey(1)
    if key==27:
        break
    if key==ord('c') and ((param[0]['class'] is not None) or (param[1]['class'] is not None)) and (mode=='train'):
        # 'c'를 누르면 라벨 변경
        # 'fist','one','two','three','four','five','six',
        class_label = (class_label + 1) % 3
        print('Change to gesture', list(gest.gesture)[class_label])
    if key==32 and (param[0]['class'] is not None) and (mode=='train'):
        # 'space bar'를 누르면 손 모양의 각을 저장 
        if (param[0]['class'] is not None) and (param[1]['class'] is None):
            if (param[0]['class'] == 'Left') and (param[1]['class'] is None):
                # 왼손만 있을 경우
                b = np.concatenate((param[0]['angle'], np.array(zeroarr)))
                print(b)
                gest.train(b, class_label)
            if (param[0]['class'] == 'Right') and (param[1]['class'] is None):
                # 오른손만 있을 경우
                b = np.concatenate((np.array(zeroarr), param[0]['angle']))
                print(b)
                gest.train(b, class_label)
        elif (param[0]['class'] is not None) and (param[1]['class'] is not None):
            # 양손일 때
            if (param[0]['class'] == 'Left'):
                b = np.concatenate((param[0]['angle'], param[1]['angle']))
                print(b)
                gest.train(b, class_label)
            if (param[0]['class'] == 'Right'):
                b = np.concatenate((param[1]['angle'], param[0]['angle']))
                print(b)
                gest.train(b, class_label)
        print('Saved', list(gest.gesture)[class_label], 'counter', counter)
        counter += 1
    if key==32 and (param[0]['class'] is not None) and (mode=='eval'):
        cv2.waitKey(0) 
        # 'esc'누르면 중단     

pipe.pipe.close()
cap.release()