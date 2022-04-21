import cv2 as cv
import numpy as np


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
        cv.rectangle(mask, (int(x/2), 0), (x, y), (255, 255, 255), -1)
        temp = cv.flip(img.copy(), 1)
        cv.copyTo(temp, mask, rst)
    elif md == 7: # 분할 화면
        grid = 3
        temp = img.copy()
        y, x = img.shape[:2]
        _x_len = int(x/grid)
        for num in range(grid):
            rst[0:y, int(num * _x_len):int((num+1) * _x_len)] = temp[0:y, int(x / 2 - _x_len / 2):int(x / 2 + _x_len / 2)]
    return rst


def filter(cam):
    mode = 1
    while True:
        ret, img_frame = cam.read()
        dst = img_frame.copy()

        dst = mode_changer(img_frame, mode)

        cv.imshow("dst", dst)
        key = cv.waitKey(1)
        if key == 27:
            break
        elif key == ord('1'):
            mode = 1
        elif key == ord('2'):
            mode = 2
        elif key == ord('3'):
            mode = 3
        elif key == ord('4'):
            mode = 4
        elif key == ord('5'):
            mode = 5
        elif key == ord('6'):
            mode = 6
        elif key == ord('7'):
            mode = 7
    cv.destroyAllWindows()

if __name__ == "__main__":
    cam = cv.VideoCapture(0 + cv.CAP_DSHOW)
    filter(cam)
