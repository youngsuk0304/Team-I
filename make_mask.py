import dlib
import cv2 as cv
import numpy as np

white_color = (255, 255, 255)

def make_mask(cam):
    while True:
        ret, img_frame = cam.read()
        h, w = img_frame.shape[:2]
        img_gray = cv.cvtColor(img_frame, cv.COLOR_BGR2GRAY)
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
    cam.release()


if __name__ == "__main__":
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')
    cap = cv.VideoCapture(0 + cv.CAP_DSHOW)

    make_mask(cap)
