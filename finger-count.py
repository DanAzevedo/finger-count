import cv2
import mediapipe as mp

cap = cv2.VideoCapture(0)

# Responsável pelas configurações do MediaPipe
hand = mp.solutions.hands
Hand = hand.Hands(max_num_hands=1)
# Desenhas as ligações entre os pontos da mão
mpDraw = mp.solutions.drawing_utils

while True:
    ret, img = cap.read()
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = Hand.process(imgRGB)
    # Extraindo onde estão os pontos que queremos identificar dentro do desenho da mão
    handsPoints = results.multi_hand_landmarks
    h, w, _ = img.shape
    array_points = []
    if handsPoints:
        for points in handsPoints:
            # print(points)
            mpDraw.draw_landmarks(img, points, hand.HAND_CONNECTIONS)
            for id, cord in enumerate(points.landmark):
                cx, cy = int(cord.x*w), int(cord.y*h)
                # cv2.putText(img, str(id), (cx, cy+10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,0,0), 2)
                array_points.append((cx, cy))
                # print(array_points)

        # Left Hand
        fingers = [8,12,16,20]
        count = 0
        if points:
            # Lógica para o polegar
            if array_points[4][0] < array_points[2][0]:
                count += 1
            # Lógica para os outros 4 dedos
            for x in fingers:
                if array_points[x][1] < array_points[x -2][1]:
                    count += 1

        cv2.rectangle(img, (80,10), (200,100), (255,0,0), -1)
        cv2.putText(img, str(count), (100,100), cv2.FONT_HERSHEY_SIMPLEX, 4, (255,255,255), 5)

    cv2.imshow('Principal', img)
    cv2.waitKey(1)

