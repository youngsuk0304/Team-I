import cv2
import dlib
import cv2 as cv
import numpy as np

DEBUG = False


def white_balance(_frame): # 화이트 밸런스
    img_yuv = cv.cvtColor(_frame, cv.COLOR_BGR2YUV)
    clahe = cv.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    img_yuv[:, :, 0] = clahe.apply(img_yuv[:, :, 0])  # CLAHE 적용
    img_yuv = cv.cvtColor(img_yuv, cv.COLOR_YUV2BGR)
    if DEBUG:
        cv.imshow("frame_clahe", img_yuv)
    return img_yuv


if __name__ == "__main__":
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')
    cap = cv.VideoCapture(0)
    #### debug
    if DEBUG:
        img_frame = cv.imread("sunju.jpg")

    while True:
        #### debug
        if not DEBUG:
            ret, img_frame = cap.read()
        img_rst = img_frame.copy() # 출력 용 이미지
        img_copy = img_frame.copy() # 원본 이미지 복사 -> img_copy

        bg_img = cv.imread('test_img.jpg')  # 배경이미지 가져오기
        x, y, w, h = 220, 120, 150, 170 # roi 영역 수동으로 지정
        bg_roi = bg_img[y:y + h, x:x + w] # roi 영역 지정

        img_gray = cv.cvtColor(white_balance(img_frame), cv.COLOR_BGR2GRAY)
        dets = detector(img_gray, 1)
        mask, rois, result = [], [], []

        for face in dets:
            shape = predictor(img_frame, face)  # 얼굴에서 68개 점 찾기
            list_points = []
            _mask = np.zeros(img_frame.shape, np.uint8)  # 원본 이미지 shape
            for p in shape.parts(): # 점 68개 리스트
                list_points.append([p.x, p.y])

            list_points = np.array(list_points)

            pt = list(range(0, 27))
            pt[17:27] = pt[26:16:-1]
            mask_pt = []
            for x in pt:
                mask_pt.append([list_points[x][0], list_points[x][1]])

            # 제로 이미지에 얼굴 위치를 잡아 백색 다각형을 생성 -> 마스킹용
            _mask = cv.fillPoly(cv.polylines(_mask, [np.int32(mask_pt)], True, (255, 255, 255)), [np.int32(mask_pt)], (255, 255, 255))
            _masked = _mask[face.top(): face.bottom(), face.left(): face.right()]

            rois = img_rst[face.top(): face.bottom(), face.left(): face.right()] # rois에 인식된 얼굴들의 각 RoI를 저장

            # 마스크로 떼어낸 얼굴을 resize 적용 필요
            rs_img = cv.resize(rois, dsize=(w, h), interpolation=cv2.INTER_LINEAR)
            mk_img = cv.resize(_masked, dsize=(w, h), interpolation=cv2.INTER_LINEAR)
            cv.copyTo(rs_img, mk_img, bg_roi)
            cv.imshow("bg_img", bg_img)
        key = cv.waitKey(30)
        if key == 27:
            break
    cap.release()
