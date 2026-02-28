import joblib
import cv2
import numpy as np

################## haar级联分类器###########################
#xml加载有两种方法
# 加载人险检测的xml(第一种方法)
face_cascade = cv2.CascadeClassifier()# 先实例化一个对象
# #这里是你的xml存放路径!!!(这个是加载人脸识别的引擎)
face_cascade.load('.\haarcascade_frontalface_default.xml')#可认为是训练好的分类器
# ## 加载人脸检测的xml(第2种方法)
#eye_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
# ##eye cascade.load('. haarcascade_frontalface_default.xml')
imgsize=64
def reSizeImg(img):
    img_gray_=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    img_gray_=cv2.resize(img_gray_,dsize=(imgsize,imgsize))
    return img_gray_.reshape(imgsize *imgsize)
def detection_face():
    svm = joblib.load('./face_svc.pkl')
    cinema =cv2.VideoCapture(0)
    count = 0
    while True:
        ret,frame =cinema.read()
        grey =cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
        faces =face_cascade.detectMultiScale(grey, scaleFactor = 1.2,
                                             minNeighbors =3,minSize=(32,32))
        #cv2.imshow('face_grey',grey)

        print("face_len=",len(faces))
        if len(faces)>0:
            for(x,y,w,h)in faces:
                face =frame[y:y+h,x:x+h]
                pre = svm.predict(reSizeImg(face).reshape(1,imgsize*imgsize))
                face_show =cv2.rectangle(frame,(x,y),(x+w,y+h),(0,0,255))
                face_show =cv2.putText(face_show,pre[0],(50,50),
                                       cv2.FONT_HERSHEY_SIMPLEX,1.2,(255,255,255),2)
                cv2.imshow('face_frame',face_show)
        if cv2.waitKey(100)& 0xff == ord('q'):
            cv2.destroyAllWindows()
            cinema.release()
            break

detection_face()

