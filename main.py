import dlib
import cv2 as cv
import numpy as np

DEBUG = False
white_color = (255, 255, 255)


def white_balance(_frame):
    img_yuv = cv.cvtColor(_frame, cv.COLOR_BGR2YUV)
    clahe = cv.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    img_yuv[:, :, 0] = clahe.apply(img_yuv[:, :, 0])  # CLAHE 적용
    img_yuv = cv.cvtColor(img_yuv, cv.COLOR_YUV2BGR)
    if DEBUG:
        cv.imshow("frame_clahe", img_yuv)
    return img_yuv


def mode_changer(img, md):
    rst = img.copy()
    if md == 2: # 색 반전
        cv.bitwise_not(img, rst)
    elif md == 3: # 좌우반전
        cv.flip(img, 1, rst)
    elif md == 4: # 상하반전
        cv.flip(img, 0, rst)
    elif md == 5: # 선화
        rst = cv.cvtColor(rst, cv.COLOR_BGR2GRAY)
        cv.bitwise_not(cv.Canny(rst, 100, 100), rst)
    elif md == 6: # 거울
        y, x = img.shape[:2]
        mask = np.zeros((y, x), np.uint8)
        cv.rectangle(mask, (int(x/2), 0), (x, y), white_color, -1)
        temp = cv.flip(img.copy(), 1)
        cv.copyTo(temp, mask, rst)
    elif md == 7: # 분할 화면
        grid = 3
        temp = img.copy()
        y, x = img.shape[:2]
        _x_len = int(x/grid)
        for num in range(grid):
            rst[0:y, int(num * _x_len):int((num+1) * _x_len)] = \
                temp[0:y, int(x / 2 - _x_len / 2):int(x / 2 + _x_len / 2)]
    return rst


def face_swap(cam):
    #### debug
    if DEBUG:
        img_frame = cv.imread("people.jpg")

    while True:
        if not DEBUG:
            rst, img_frame = cam.read()
        img_rst = img_frame.copy()  # 출력 용 이미지
        img_copy = img_frame.copy()  # 원본 이미지 복사 -> img_copy

        img_gray = cv.cvtColor(white_balance(img_frame), cv.COLOR_BGR2GRAY)
        dets = detector(img_gray, 1)
        mask, rois, result = [], [], []

        for face in dets:
            shape = predictor(img_frame, face)  # 얼굴에서 68개 점 찾기
            list_points = []
            _mask = np.zeros(img_frame.shape, np.uint8)
            for p in shape.parts():  # 점 68개 리스트
                list_points.append([p.x, p.y])

            list_points = np.array(list_points)

            pt = list(range(0, 27))
            pt[17:27] = pt[26:16:-1]  # 얼굴 윤곽선 점
            mask_pt = []
            for x in pt:
                mask_pt.append([list_points[x][0], list_points[x][1]])

            # 제로 이미지에 얼굴 위치를 잡아 백색 다각형을 생성 -> 마스킹용
            _mask = cv.fillPoly(cv.polylines(_mask, [np.int32(mask_pt)], True, white_color), [np.int32(mask_pt)], white_color)
            _masked = _mask[face.top(): face.bottom(), face.left(): face.right()]
            mask.append(_masked.copy())

            rois.append(img_rst[face.top(): face.bottom(), face.left(): face.right()])  # rois에 인식된 얼굴들의 각 RoI를 저장
            result.append(img_copy[face.top(): face.bottom(), face.left(): face.right()])  # 얼굴 합성할 베이스
        #### debug
        if DEBUG:
            for i in range(len(rois)):
                cv.imshow('rois' + str(i), rois[i])
                cv.imshow("mask" + str(i), mask[i])

        if len(rois) > 1:  # 얼굴이 둘 이상 인식되었을 때
            for i in range(len(rois)):
                j = i + 1
                if i == len(rois) - 1:
                    j = 0
                temp = rois[i].copy()
                temp = cv.resize(temp, dsize=(rois[j].shape[1], rois[j].shape[0]))
                _mask = mask[i].copy()
                _mask = cv.resize(_mask, dsize=(rois[j].shape[1], rois[j].shape[0]))
                cv.copyTo(temp, _mask, result[j])
        cv.imshow('result', img_copy)
        key = cv.waitKey(1)
        if key == 27:
            break
    cv.destroyAllWindows()


def make_mask(cam):
    while True:
        rst, img_frame = cam.read()
        h, w = img_frame.shape[:2]
        img_gray = cv.cvtColor(white_balance(img_frame), cv.COLOR_BGR2GRAY)
        dets = detector(img_gray, 1)

        for face in dets:
            shape = predictor(img_frame, face)  # 얼굴에서 68개 점 찾기
            list_points = []
            #포인트 좌표
            for p in shape.parts():
                list_points.append([p.x, p.y])
            list_points = np.array(list_points)

            pt = list(range(2, 15))  # 마스크 위치 점
            pt.append(28)
            pt[:] = pt[::-1]
            mask_pt = []
            for x in pt:
                mask_pt.append([list_points[x][0], list_points[x][1]])

            img_frame = cv.fillConvexPoly(img_frame, np.array([mask_pt], np.int32), white_color)
        cv.flip(img_frame, 1, img_frame)
        cv.imshow('result', img_frame)
        key = cv.waitKey(1)
        if key == 27:
            break
    cv.destroyAllWindows()


def filter(cam):
    _mode = 1
    while True:
        rst, img_frame = cam.read()
        dst = img_frame.copy()

        dst = mode_changer(img_frame, _mode)

        cv.imshow("dst", dst)
        _key = cv.waitKey(1)
        if _key == 27:
            break
        elif _key == ord('1'):
            _mode = 1
        elif _key == ord('2'):
            _mode = 2
        elif _key == ord('3'):
            _mode = 3
        elif _key == ord('4'):
            _mode = 4
        elif _key == ord('5'):
            _mode = 5
        elif _key == ord('6'):
            _mode = 6
        elif _key == ord('7'):
            _mode = 7
    cv.destroyAllWindows()


def face_background(cam):
    while True:
        #### debug
        if not DEBUG:
            ret, img_frame = cam.read()
        img_rst = img_frame.copy()  # 출력 용 이미지
        img_copy = img_frame.copy()  # 원본 이미지 복사 -> img_copy

        bg_img = cv.imread('test_img.jpg')  # 배경이미지 가져오기
        # roi 영역 수동으로 지정
        x, y, w, h = 220, 120, 150, 170
        bg_roi = bg_img[y:y + h, x:x + w]  # roi 영역 지정

        img_gray = cv.cvtColor(white_balance(img_frame), cv.COLOR_BGR2GRAY)
        dets = detector(img_gray, 1)
        mask, rois, result = [], [], []

        for face in dets:
            shape = predictor(img_frame, face)  # 얼굴에서 68개 점 찾기
            list_points = []
            _mask = np.zeros(img_frame.shape, np.uint8)  # 원본 이미지 shape

            for i, p in enumerate(shape.parts()):  # 점 68개 리스트
                if 17<=i<=26:
                    p.y-=20
                if i==20 or i==23:
                    p.y-=20
                elif i==21 or i ==22:
                    p.y-=30
                list_points.append([p.x, p.y])

            list_points = np.array(list_points)

            pt = list(range(0, 27))
            pt[17:27] = pt[26:16:-1]
            mask_pt = []
            for x in pt:
                mask_pt.append([list_points[x][0], list_points[x][1]])

            # 제로 이미지에 얼굴 위치를 잡아 백색 다각형을 생성 -> 마스킹용
            _mask = cv.fillPoly(cv.polylines(_mask, [np.int32(mask_pt)], True, white_color), [np.int32(mask_pt)], white_color)
            _masked = _mask[face.top(): face.bottom(), face.left(): face.right()]
            rois = img_rst[face.top(): face.bottom(), face.left(): face.right()]  # rois에 인식된 얼굴들의 각 RoI를 저장

            # 마스크로 떼어낸 얼굴을 resize 적용 필요
            rs_img = cv.resize(rois, dsize=(w, h), interpolation=cv.INTER_LINEAR)
            mk_img = cv.resize(_masked, dsize=(w, h), interpolation=cv.INTER_LINEAR)
            cv.copyTo(rs_img, mk_img, bg_roi)
            cv.imshow("bg_img", bg_img)
        key = cv.waitKey(30)
        if key == 27:
            break
    cv.destroyAllWindows()


if __name__ == "__main__":
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')
    cap = cv.VideoCapture(0 + cv.CAP_DSHOW)

    while True:
        ret, img_frame = cap.read()
        cv.flip(img_frame, 1, img_frame)
        cv.imshow("waiting", img_frame)

        key = cv.waitKey(1)
        if key == 27:
            break
        elif key == ord('2'):
            cv.destroyAllWindows()
            face_swap(cap)
        elif key == ord('3'):
            cv.destroyAllWindows()
            make_mask(cap)
        elif key == ord('4'):
            cv.destroyAllWindows()
            filter(cap)
        elif key == ord('5'):
            cv.destroyAllWindows()
            face_background(cap)
    cap.release()