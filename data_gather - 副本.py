import numpy as np
import os
import cv2  #pip install opencv-python
import time


face_cascade = cv2.CascadeClassifier()

face_cascade.load('.\haarcascade_frontalface_default.xml')

eye_cascade = cv2.CascadeClassifier('.\haarcascade_eye.xml')

from getImgData import GetImgData
getdata = GetImgData()  # 图片获取类


class CaptureFace:
    def __init__(self,imgdir='./test_img/',grayfacedir='./test_img_gray'):

        self.imgdir = imgdir
        self.grayfacedir = grayfacedir

    def captureface(self,someone=None,picturenum=10,waitkey=300):
        """
        利用opencv调用电脑摄像头来拍照，并进行存储
        """
        filepath = os.path.join(self.imgdir, someone)
        if not os.path.exists(filepath):
            os.makedirs(filepath)
        capture = cv2.VideoCapture(0)
        time.sleep(1)
        for i in range(picturenum):
            _,img = capture.read()
            cv2.imshow(someone, img)
            if cv2.waitKey(waitkey) == ord('q'):
                break
            picturepath = os.path.join(filepath,str(i))+'.jpg'
            cv2.imwrite(picturepath,img,[cv2.IMWRITE_JPEG_QUALITY,100])
        capture.release()
        cv2.destroyAllWindows()
        print('done')

    def facetogray(self,someone='zhangmin',size=64,waitkey=100):
        print("self.imgdir=", self.imgdir)
        imgnames = getdata.getimgnames(path=os.path.join(self.imgdir,someone))
        n=len(imgnames)
        print("n=", n)
        newpath = os.path.join(self.grayfacedir,someone)
        if not os.path.exists(newpath):
            os.makedirs(newpath)
        for i in range(n):
            img = cv2.imread(imgnames[i])
            # 开始人脸检测
            print("img:", img)
            faces = face_cascade.detectMultiScale(img, scaleFactor=1.3, minNeighbors=5)
            print("faces:",faces)
            # 在检测到的人脸中操作
            face_area = None
            for x, y, w, h in faces:
                img2 = cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)

                face_area = img2[y:y + h, x:x + w]

            if((face_area is not None) and len(face_area) > 0):
                face_gray = cv2.cvtColor(face_area, cv2.COLOR_BGR2GRAY)
                face_gray = cv2.resize(face_gray, (size, size))
                cv2.imshow('image', face_gray)
                cv2.imwrite(newpath + '/' + str(i) + '.jpg', face_gray)
                if cv2.waitKey(waitkey) == ord('q'):
                    break
        cv2.destroyAllWindows()

if __name__ == '__main__':
    picture = CaptureFace()
    picture.captureface(picturenum=100,waitkey=200,someone='你的人脸数据库')
    picture.facetogray(someone='你的人脸数据库',size=64,waitkey=500)

