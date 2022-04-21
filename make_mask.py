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
        list_points = np.array(list_points)

        img_frame= cv.fillConvexPoly(img_frame, np.array([[list_points[28][0], list_points[28][1]], [list_points[14][0], list_points[14][1]], [list_points[13][0], list_points[13][1]],
                                                          [list_points[12][0], list_points[12][1]], [list_points[11][0], list_points[11][1]], [list_points[10][0], list_points[10][1]],
                                                          [list_points[9][0], list_points[9][1]], [list_points[8][0], list_points[8][1]], [list_points[7][0], list_points[7][1]],
                                                          [list_points[6][0], list_points[6][1]], [list_points[5][0], list_points[5][1]], [list_points[4][0], list_points[4][1]],
                                                          [list_points[3][0], list_points[3][1]], [list_points[2][0], list_points[2][1]]], np.int32), white_color)
        #얼굴 마스크 생성과 원하는 이미지에 copyTo해서 씌우기
        '''
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
        '''
    #cv.imshow("img", img)
    cv.flip(img_frame,1,img_frame)
    cv.imshow('result', img_frame)
    key = cv.waitKey(1)

    if key == 27:
        break

cap.release()

