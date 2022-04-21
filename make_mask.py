# import cv2
#
# font = cv2.FONT_ITALIC
#
#
# def faceDetect():
#     eye_detect = False
#     face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")  # 얼굴찾기 haar 파일
#     eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_eye.xml")  # 눈찾기 haar 파일
#     try:
#         cam = cv2.VideoCapture(0)
#     except:
#         print("camera loading error")
#         return
#
#     while True:
#         ret, frame = cam.read()
#         if not ret:
#             break
#
#         if eye_detect:
#             info = "Eye Detention ON"
#         else:
#             info = "Eye Detection OFF"
#
#         gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#         faces = face_cascade.detectMultiScale(gray, 1.3, 5)
#
#         faces = face_cascade.detectMultiScale(gray, 1.3, 5)
#         # 카메라 영상 왼쪽위에 위에 셋팅된 info 의 내용 출력
#         cv2.putText(frame, info, (5, 15), font, 0.5, (255, 0, 255), 1)
#
#         for (x, y, w, h) in faces:
#             cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)  # 사각형 범위
#             cv2.putText(frame, "Detected Face", (x - 5, y - 5), font, 0.5, (255, 255, 0), 2)  # 얼굴찾았다는 메시지
#             if eye_detect:  #찾 눈기
#                 roi_gray = gray[y:y + h, x:x + w]
#                 roi_color = frame[y:y + h, x:x + w]
#                 eyes = eye_cascade.detectMultiScale(roi_gray)
#                 for (ex, ey, ew, eh) in eyes:
#                     cv2.rectangle(roi_color, (ex, ey), (ex + ew, ey + eh), (0, 255, 0), 2)
#
#         cv2.flip(frame,1,frame)
#         cv2.imshow("frame", frame)
#         k = cv2.waitKey(30)
#         if k == ord('i'):
#             eye_detect = not eye_detect
#         if k == 27:
#             break
#         # 실행 중 키보드 i 를 누르면 눈찾기를 on, off한다.
#
#     cam.release()
#     cv2.destroyAllWindows()
#
# faceDetect()
# 패키지 설치
# pip install dlib opencv-python
#
# 학습 모델 다운로드
# http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2
import dlib
import cv2 as cv
import numpy as np

detector = dlib.get_frontal_face_detector()

predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')

cap = cv.VideoCapture(0+cv.CAP_DSHOW)

# range는 끝값이 포함안됨
ALL = list(range(0, 68))
RIGHT_EYEBROW = list(range(17, 22))
LEFT_EYEBROW = list(range(22, 27))
RIGHT_EYE = list(range(36, 42))
LEFT_EYE = list(range(42, 48))
NOSE = list(range(27, 36))
MOUTH_OUTLINE = list(range(48, 61))
MOUTH_INNER = list(range(61, 68))
JAWLINE = list(range(0, 17))
TEST = list([2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 28])
test = np.array([28, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2], np.int32)
index = ALL
white_color=(255,255,255)
while True:

    ret, img_frame = cap.read()
    h,w,c=img_frame.shape
    img = cv.imread('Test.jpg')
    img = cv.resize(img, (w, h), interpolation=cv.INTER_AREA)
    img_gray = cv.cvtColor(img_frame, cv.COLOR_BGR2GRAY)

    dets = detector(img_gray, 1)

    for face in dets:

        shape = predictor(img_frame, face)  # 얼굴에서 68개 점 찾기

        list_points = []
        #포인트 좌표
        for p in shape.parts():
            list_points.append([p.x, p.y])
            #print(p.x+"         "+p.y)

        list_points = np.array(list_points)
        '''
        for i, pt in enumerate(list_points[index]):
            pt_pos = (pt[0], pt[1])
            cv.circle(img_frame, pt_pos, 2, (0, 255, 0), -1)
        '''

        #cv.rectangle(img_frame, (face.left(), face.top()), (face.right(), face.bottom()), (0, 0, 255), 3)
        '''
        img_frame= cv.fillConvexPoly(img_frame, np.array([[list_points[28][0], list_points[28][1]], [list_points[14][0], list_points[14][1]], [list_points[13][0], list_points[13][1]],
                                                          [list_points[12][0], list_points[12][1]], [list_points[11][0], list_points[11][1]], [list_points[10][0], list_points[10][1]],
                                                          [list_points[9][0], list_points[9][1]], [list_points[8][0], list_points[8][1]], [list_points[7][0], list_points[7][1]],
                                                          [list_points[6][0], list_points[6][1]], [list_points[5][0], list_points[5][1]], [list_points[4][0], list_points[4][1]],

                                                          [list_points[3][0], list_points[3][1]], [list_points[2][0], list_points[2][1]]], np.int32), white_color)
        '''

        # roi=img_frame[np.array(
        #     [[list_points[14][0], list_points[14][1]], [list_points[13][0], list_points[13][1]], [list_points[12][0], list_points[12][1]],
        #     [list_points[11][0], list_points[11][1]], [list_points[10][0], list_points[10][1]], [list_points[9][0], list_points[9][1]],
        #     [list_points[8][0], list_points[8][1]], [list_points[7][0], list_points[7][1]], [list_points[6][0], list_points[6][1]],
        #     [list_points[5][0], list_points[5][1]], [list_points[4][0], list_points[4][1]], [list_points[3][0], list_points[3][1]],
        #     [list_points[2][0], list_points[2][1]], [list_points[3][0], list_points[3][1]], [list_points[2][0], list_points[2][1]],
        #     [list_points[1][0], list_points[1][1]], [list_points[0][0], list_points[0][1]-50], [list_points[17][0], list_points[17][1]-50],
        #     [list_points[18][0], list_points[18][1]-50], [list_points[19][0], list_points[19][1]-60], [list_points[20][0], list_points[20][1]-70],
        #     [list_points[21][0], list_points[21][1]-80], [list_points[22][0], list_points[22][1]-80], [list_points[23][0], list_points[23][1]-70],
        #     [list_points[24][0], list_points[24][1]-60], [list_points[25][0], list_points[25][1]-50], [list_points[26][0], list_points[26][1]-50],
        #     [list_points[16][0], list_points[16][1]-50], [list_points[15][0], list_points[15][1]]])]

        # cv.polylines(roi, np.int32([np.array(
        #     [[list_points[14][0], list_points[14][1]], [list_points[13][0], list_points[13][1]],
        #      [list_points[12][0], list_points[12][1]],
        #      [list_points[11][0], list_points[11][1]], [list_points[10][0], list_points[10][1]],
        #      [list_points[9][0], list_points[9][1]],
        #      [list_points[8][0], list_points[8][1]], [list_points[7][0], list_points[7][1]],
        #      [list_points[6][0], list_points[6][1]],
        #      [list_points[5][0], list_points[5][1]], [list_points[4][0], list_points[4][1]],
        #      [list_points[3][0], list_points[3][1]],
        #      [list_points[2][0], list_points[2][1]], [list_points[3][0], list_points[3][1]],
        #      [list_points[2][0], list_points[2][1]],
        #      [list_points[1][0], list_points[1][1]], [list_points[0][0], list_points[0][1] - 50],
        #      [list_points[17][0], list_points[17][1] - 50],
        #      [list_points[18][0], list_points[18][1] - 50], [list_points[19][0], list_points[19][1] - 60],
        #      [list_points[20][0], list_points[20][1] - 70],
        #      [list_points[21][0], list_points[21][1] - 80], [list_points[22][0], list_points[22][1] - 80],
        #      [list_points[23][0], list_points[23][1] - 70],
        #      [list_points[24][0], list_points[24][1] - 60], [list_points[25][0], list_points[25][1] - 50],
        #      [list_points[26][0], list_points[26][1] - 50],
        #      [list_points[16][0], list_points[16][1] - 50], [list_points[15][0], list_points[15][1]]])]), True,
        #              white_color, 1)
        '''cv.polylines(img_frame, np.int32([np.array(
            [[list_points[14][0], list_points[14][1]], [list_points[13][0], list_points[13][1]], [list_points[12][0], list_points[12][1]],
            [list_points[11][0], list_points[11][1]], [list_points[10][0], list_points[10][1]], [list_points[9][0], list_points[9][1]],
            [list_points[8][0], list_points[8][1]], [list_points[7][0], list_points[7][1]], [list_points[6][0], list_points[6][1]],
            [list_points[5][0], list_points[5][1]], [list_points[4][0], list_points[4][1]], [list_points[3][0], list_points[3][1]],
            [list_points[2][0], list_points[2][1]], [list_points[3][0], list_points[3][1]], [list_points[2][0], list_points[2][1]],
            [list_points[1][0], list_points[1][1]], [list_points[0][0], list_points[0][1]-50], [list_points[17][0], list_points[17][1]-50],
            [list_points[18][0], list_points[18][1]-50], [list_points[19][0], list_points[19][1]-60], [list_points[20][0], list_points[20][1]-70],
            [list_points[21][0], list_points[21][1]-80], [list_points[22][0], list_points[22][1]-80], [list_points[23][0], list_points[23][1]-70],
            [list_points[24][0], list_points[24][1]-60], [list_points[25][0], list_points[25][1]-50], [list_points[26][0], list_points[26][1]-50],
            [list_points[16][0], list_points[16][1]-50], [list_points[15][0], list_points[15][1]]])]), True, white_color, 1)
        '''
        '''img_test_frame = img_frame(cv.polylines(img_frame, [np.array(
            [[list_points[14][0], list_points[14][1]], [list_points[13][0], list_points[13][1]],
             [list_points[12][0], list_points[12][1]], [list_points[11][0], list_points[11][1]],
             [list_points[10][0], list_points[10][1]], [list_points[9][0], list_points[9][1]],
             [list_points[8][0], list_points[8][1]], [list_points[7][0], list_points[7][1]],
             [list_points[6][0], list_points[6][1]], [list_points[5][0], list_points[5][1]],
             [list_points[4][0], list_points[4][1]], [list_points[3][0], list_points[3][1]],
             [list_points[2][0], list_points[2][1]], [list_points[3][0], list_points[3][1]],
             [list_points[2][0], list_points[2][1]], [list_points[1][0], list_points[1][1]],
             [list_points[0][0], list_points[0][1]], [list_points[17][0], list_points[17][1]],
             [list_points[18][0], list_points[18][1]], [list_points[19][0], list_points[19][1]],
             [list_points[20][0], list_points[20][1]], [list_points[21][0], list_points[21][1]],
             [list_points[22][0], list_points[22][1]], [list_points[23][0], list_points[23][1]],
             [list_points[24][0], list_points[24][1]], [list_points[25][0], list_points[25][1]],
             [list_points[26][0], list_points[26][1]], [list_points[16][0], list_points[16][1]],
             [list_points[15][0], list_points[15][1]]],np.int32)], True, white_color,1))'''

        mask = np.zeros(img_frame.shape, dtype=np.uint8)
        roi_corners = np.int32([np.array(
            [np.array(
                [[list_points[14][0], list_points[14][1]], [list_points[13][0], list_points[13][1]],
                 [list_points[12][0], list_points[12][1]],
                 [list_points[11][0], list_points[11][1]], [list_points[10][0], list_points[10][1]],
                 [list_points[9][0], list_points[9][1]],
                 [list_points[8][0], list_points[8][1]], [list_points[7][0], list_points[7][1]],
                 [list_points[6][0], list_points[6][1]],
                 [list_points[5][0], list_points[5][1]], [list_points[4][0], list_points[4][1]],
                 [list_points[3][0], list_points[3][1]],
                 [list_points[2][0], list_points[2][1]], [list_points[3][0], list_points[3][1]],
                 [list_points[2][0], list_points[2][1]],
                 [list_points[1][0], list_points[1][1]], [list_points[0][0], list_points[0][1] - 50],
                 [list_points[17][0], list_points[17][1] - 50],
                 [list_points[18][0], list_points[18][1] - 50], [list_points[19][0], list_points[19][1] - 60],
                 [list_points[20][0], list_points[20][1] - 70],
                 [list_points[21][0], list_points[21][1] - 80], [list_points[22][0], list_points[22][1] - 80],
                 [list_points[23][0], list_points[23][1] - 70],
                 [list_points[24][0], list_points[24][1] - 60], [list_points[25][0], list_points[25][1] - 50],
                 [list_points[26][0], list_points[26][1] - 50],
                 [list_points[16][0], list_points[16][1] - 50], [list_points[15][0], list_points[15][1]]])])])
        channel_count = img_frame.shape[2]
        ignore_mask_color = (255,) * channel_count
        cv.fillPoly(mask, roi_corners, ignore_mask_color)

        cv.copyTo(img_frame, mask, img)
        masked_image = cv.bitwise_and(img_frame, mask)

        cv.imshow("test", masked_image)
        #cv.flip(img, 1, img)

    cv.imshow("img", img)

    cv.flip(img_frame,1,img_frame)

    cv.imshow('result', img_frame)

    key = cv.waitKey(1)

    if key == 27:
        break

    elif key == ord('1'):
        index = ALL
    elif key == ord('2'):
        index = LEFT_EYEBROW + RIGHT_EYEBROW
    elif key == ord('3'):
        index = LEFT_EYE + RIGHT_EYE
    elif key == ord('4'):
        index = NOSE
    elif key == ord('5'):
        index = MOUTH_OUTLINE + MOUTH_INNER
    elif key == ord('6'):
        index = JAWLINE
    elif key == ord('7'):
        index = TEST
    elif key == ord('8'):
        index = LEFT_EYEBROW + RIGHT_EYEBROW + JAWLINE

cap.release()

