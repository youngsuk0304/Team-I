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


def face_swap(cam):
    #### debug
    if DEBUG:
        img_frame = cv.imread("people.jpg")

    while True:
        #### debug
        if not DEBUG:
            ret, img_frame = cap.read()
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
            _mask = cv.fillPoly(cv.polylines(_mask, [np.int32(mask_pt)], True, (255, 255, 255)), [np.int32(mask_pt)],
                                (255, 255, 255))
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
    cam.release()


if __name__ == "__main__":
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')
    cap = cv.VideoCapture(0 + cv.CAP_DSHOW)

    face_swap(cap)
