import cv2
import mediapipe as mp
import numpy as np
import time, os
import csv

label_file = open('label.csv','r')
csv_read = csv.reader(label_file)

time_length = [0 for i in range(7500)]
for i, line in enumerate(csv_read, 0):
    time_length[i] = float(line[0])

actions = list(range(1,1501))

seq_length = 30

# MediaPipe hands model
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(max_num_hands=2, min_detection_confidence=0.5, min_tracking_confidence=0.5)

os.makedirs('dataset', exist_ok=True)

for i, filename in enumerate(os.listdir("D:/data")):
    print(filename)
    cap = cv2.VideoCapture(os.path.join("D:/data", filename))
    secs_for_action = time_length[i // 5 + 1]
    while cap.isOpened():
        for idx, action in enumerate(actions):
            data = []

            ret, img = cap.read()

            if not ret:
                break

            img = cv2.flip(img, 1)

            cv2.imshow('img', img)

            start_time = time.time()

            while time.time() - start_time < secs_for_action:
                ret, img = cap.read()
                if not ret:
                    break

                img = cv2.flip(img, 1)
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                result = hands.process(img)
                img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

                if result.multi_hand_landmarks is not None:
                    for res in result.multi_hand_landmarks:
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
                        angle_label = np.append(angle_label, idx)

                        d = np.concatenate([joint.flatten(), angle_label])

                        data.append(d)

                        mp_drawing.draw_landmarks(img, res, mp_hands.HAND_CONNECTIONS)

                cv2.imshow('img', img)
                if cv2.waitKey(1) == ord('q'):
                    break

            data = np.array(data)
            print(action // 5 + 1, data.shape)
            np.save(os.path.join('dataset', f'raw_{action//5 + 1}_{action}'), data)

            # Create sequence data
            full_seq_data = []
            for seq in range(len(data) - seq_length):
                full_seq_data.append(data[seq:seq + seq_length])

            full_seq_data = np.array(full_seq_data)
            np.save(os.path.join('dataset', f'seq_{action//5 + 1}_{action}'), full_seq_data)
        break