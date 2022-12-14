import cv2
import mediapipe as mp
import numpy as np
import time, os
import csv
import sys

label_file = open('label.csv','r')
csv_read = csv.reader(label_file)

actions = list(range(1501))

seq_length = 30

# MediaPipe hands model
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(max_num_hands=2, min_detection_confidence=0.5, min_tracking_confidence=0.5)

os.makedirs('dataset', exist_ok=True)

for i, filename in enumerate(os.listdir("D:/data")):
    print(filename)
    cap = cv2.VideoCapture(os.path.join("D:/data", filename))

    while cap.isOpened():
        ret, img = cap.read()

        if not ret:
            break

        img = cv2.flip(img, 1)

        cv2.imshow('img', img)

        start_time = time.time()

        data = []
        while (1):
            ret, img = cap.read()
            if not ret:
                break

            img = cv2.flip(img, 1)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            result = hands.process(img)
            print(result)
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

            if result.multi_hand_landmarks is not None:
                for res in result.multi_hand_landmarks:
                    print(res.landmark.length)
                    sys.exit()
                    joint = np.zeros((21, 4))
                    for j, lm in enumerate(res.landmark):
                        joint[j] = [lm.x, lm.y, lm.z, lm.visibility]

                    # Compute angles between joints
                    v1 = joint[[0,1,2,3,0,5,6,7,0,9,10,11,0,13,14,15,0,17,18,19], :3] # Parent joint
                    v2 = joint[[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20], :3] # Child joint
                    v = v2 - v1 # [20, 3]
                    # Normalize v
                    v = v / np.linalg.norm(v, axis=1)[:, np.newaxis]

                    # Get angle using arcos of dot product
                    angle = np.arccos(np.einsum('nt,nt->n',
                        v[[0,1,2,4,5,6,8,9,10,12,13,14,16,17,18],:], 
                        v[[1,2,3,5,6,7,9,10,11,13,14,15,17,18,19],:])) # [15,]

                    angle = np.degrees(angle) # Convert radian to degree

                    angle_label = np.array([angle], dtype=np.float32)
                    angle_label = np.append(angle_label, actions[i] //5 +1)

                    d = np.concatenate([joint.flatten(), angle_label])

                    data.append(d)

                    mp_drawing.draw_landmarks(img, res, mp_hands.HAND_CONNECTIONS)

            cv2.imshow('img', img)
            if cv2.waitKey(1) == ord('q'):
                sys.exit()

        data = np.array(data)
        print(actions[i] // 5 + 1, data.shape)
        np.save(os.path.join('./dataset', f'raw_{actions[i] //5 + 1}_{actions[i] % 5 + 1}'), data)

        # Create sequence data
        full_seq_data = []
        for seq in range(len(data) - seq_length):
            full_seq_data.append(data[seq:seq + seq_length])

        full_seq_data = np.array(full_seq_data)
        np.save(os.path.join('./dataset', f'seq_{actions[i] //5 + 1}_{actions[i] % 5 + 1}'), full_seq_data)