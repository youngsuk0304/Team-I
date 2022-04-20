import dlib
import cv2 as cv
import numpy as np

DEBUG = True


def white_balance(_frame):
    img_yuv = cv.cvtColor(_frame, cv.COLOR_BGR2YUV)
    clahe = cv.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    img_yuv[:, :, 0] = clahe.apply(img_yuv[:, :, 0])  # CLAHE 적용
    img_yuv = cv.cvtColor(img_yuv, cv.COLOR_YUV2BGR)

    if DEBUG:
        cv.imshow("frame_clahe", img_yuv)
    return img_yuv


detector = dlib.get_frontal_face_detector()

predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')

cap = cv.VideoCapture(0)

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
MASKLINE = list((2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 28))
NOSELINE = list((29, 31, 32, 33, 34, 35, 36))
FACE_OUTLINE = list(range(0, 28))

index = ALL
if DEBUG:
    img_frame = cv.resize(cv.imread("people.jpg"), (0,0), fx=0.4, fy=0.4)

while True:
    if not DEBUG:
        ret, img_frame = cap.read()
    img_pred = img_frame.copy() # 출력 용 이미지
    img_rst = img_frame.copy() # 출력 용 이미지
    img_copy = img_frame.copy() # 원본 이미지 복사 -> img_copy

    img_gray = cv.cvtColor(white_balance(img_frame), cv.COLOR_BGR2GRAY)
    dets = detector(img_gray, 1)
    roi = np.zeros(img_frame.shape, np.uint8)
    mask = []
    rois =[]

    for face in dets:
        shape = predictor(img_frame, face)  # 얼굴에서 68개 점 찾기
        list_points = []
        _mask = np.zeros(img_frame.shape, np.uint8)
        for p in shape.parts(): # 점 68개 리스트
            list_points.append([p.x, p.y])

        list_points = np.array(list_points)

        for i, pt in enumerate(list_points[index]):
            pt_pos = (pt[0], pt[1])
            cv.circle(img_pred, pt_pos, 2, (0, 255, 0), -1)
        cv.rectangle(img_pred, (face.left(), face.top()), (face.right(), face.bottom()), (0, 0, 255), 3)
        # img_frame에 얼굴 점들과 네모가 그려져 있다.
        ####
        test = np.array(
        [[list_points[14][0], list_points[14][1]], [list_points[13][0], list_points[13][1]],
         [list_points[12][0], list_points[12][1]], [list_points[11][0], list_points[11][1]],
         [list_points[10][0], list_points[10][1]], [list_points[9][0], list_points[9][1]],
         [list_points[8][0], list_points[8][1]], [list_points[7][0], list_points[7][1]],
         [list_points[6][0], list_points[6][1]], [list_points[5][0], list_points[5][1]],
         [list_points[4][0], list_points[4][1]], [list_points[3][0], list_points[3][1]],
         [list_points[2][0], list_points[2][1]], [list_points[1][0], list_points[1][1]],
         [list_points[0][0], list_points[0][1]], [list_points[17][0], list_points[17][1]],
         [list_points[18][0], list_points[18][1]], [list_points[19][0], list_points[19][1]],
         [list_points[20][0], list_points[20][1]], [list_points[21][0], list_points[21][1]],
         [list_points[22][0], list_points[22][1]], [list_points[23][0], list_points[23][1]],
         [list_points[24][0], list_points[24][1]], [list_points[25][0], list_points[25][1]],
         [list_points[26][0], list_points[26][1]], [list_points[16][0], list_points[16][1]],
         [list_points[15][0], list_points[15][1]]], np.int32)

        # 제로 이미지에 얼굴 위치를 잡아 백색 다각형을 생성 -> 마스킹용
        _mask = cv.fillPoly(cv.polylines(_mask, [test], True, (255, 255, 255)), [test], (255, 255, 255))
        # cv.threshold(_mask, 254, 255, cv.THRESH_BINARY, _mask)
        # ret = cv.bitwise_and(_mask, img_copy)
        _masked = _mask[face.top(): face.bottom(), face.left(): face.right()]
        # rois.append(ret.copy())
        mask.append(_masked.copy())

        ret = img_rst[face.top(): face.bottom(), face.left(): face.right()]
        rois.append(ret.copy()) # rois에 인식된 얼굴들의 각 RoI를 저장

    # for i in range(len(rois)):
    #     cv.imshow('rois'+str(i), rois[i])
    #     cv.imshow("mask"+str(i), mask[i])

    if len(rois) > 1: # 얼굴이 둘 이상 인식되었을 때
        for i in range(len(rois)):
            j = i + 1
            if i == len(rois)-1:
                j = 0
            temp = rois[i].copy()
            temp = cv.resize(temp, dsize=(rois[j].shape[:2]))
            _mask = mask[i].copy()
            _mask = cv.resize(_mask, dsize=(rois[j].shape[:2]))
            ret = cv.bitwise_and(_mask, temp)
            # cv.copyTo(ret, rois[j])
            cv.imshow("re", ret)
            cv.imshow("ro", rois[j])
    #### debug
    if DEBUG:
        cv.imshow("dot", cv.resize(img_pred, (0, 0), fx=0.5, fy=0.5))
    cv.imshow('result', img_rst)
    key = cv.waitKey(30)
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
        index = MASKLINE
    elif key == ord('8'):
        index = LEFT_EYE + RIGHT_EYE + MOUTH_OUTLINE + NOSELINE

cap.release()
