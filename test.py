import tkinter
import joblib
import cv2
import numpy as np

#############haar级联分类器并#####并##并#并##并#挂林#并#并挂#并##挂#并并#并#并并#并##了
#xml加载有两种方法
# 加载人脸检测的xml(第一种方法)
face_cascade =cv2.CascadeClassifier() #先实例化一个对象
# 这里是你的xml存放路径!!!(这个是加载人脸识别的引擎)
face_cascade.load('.\haarcascade_frontalface_default.xml')#可认为是训练好的分类器
# # 加载人脸检测的xml(第2种方法)
# eye cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
# # eye cascade.load('.\haarcascade_frontalface_default.xml')
imgshow_size = 400 #用来显示的尺寸
imgsize=64 #用来计算的尺寸大小
def reSizeImg(img):
    img_gray_ = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img_gray_ = cv2.resize(img_gray_,dsize=(imgsize, imgsize))
    return img_gray_.reshape(imgsize *imgsize)

def detection_face():
    # 创建两个 Label 用于显示图片
    label1 = tk.Label(root,width=400)
    label2 = tk.Label(root,width=400)
    label1.pack(side=tk.LEFT)
    label2.pack(side=tkinter.RIGHT)

    svm = joblib.load('./face_svc.pkl')
    cinema =cv2.VideoCapture(0)
    count = 0
    while True:
        ret, frame = cinema.read()
        grey = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(grey, scaleFactor = 1.2,
                                              minNeighbors=3,minSize =(32,32))
        ## 显示器摄像头采集的人脸
        pil_img1 = Image.fromarray(grey)
        pil_img1 = pil_img1.resize((imgshow_size,imgshow_size))
        photo1 = ImageTk.PhotoImage(image=pil_img1)
        label1.config(image=photo1)
        label1.image = photo1

        print("face len=",len(faces))
        if len(faces) > 0:
            for(x,y,w,h) in faces:
                face = frame[y:y+h,x:x+h]
    #            pre = knn.predict(yu1(face).reshape(1, 24025))
    #            svm预测
                pre =svm.predict(reSizeImg(face).reshape(1,imgsize*imgsize))
                face_show =cv2.rectangle(frame,(x,y),(x+w,y+h),(0,0,255))
                face_show = cv2.putText(face_show,pre[0],(50,50),
                                        cv2.FONT_HERSHEY_SIMPLEX,1.2,(255,255,255),2)
            ## 显示检测到的人脸图片
            pil_img2 = Image.fromarray(face_show)   #PIL.Image.fromarray()将array转换为image
            pil_img2 = pil_img2.resize((imgshow_size,imgshow_size)) #调整图片大小
            photo2 = ImageTk.PhotoImage(image=pil_img2)   #将image转换为photoimage
            label2.config(image=photo2)  #将Photoimage显示在label上
            label2.image = photo2 #防止label被垃圾回收

            # cv2.imshow('face_frame',face_show)
        # if cv2.waitKey(100)& 0xff == ord('q'):
        #     cv2.destroyAllWindows()
        #     cinema.release()
        #     break

# 创建主窗口
import tkinter as tk
from PIL import Image,ImageTk

root = tk.Tk()
# 设置窗口的宽度和高度，格式为“宽度x高度”
root.geometry("800x400")
root.title("人脸检测")

# 启动主循环
if __name__== "__main__":
    #创建线程，接收客户消息并回应消息
    import threading # 导入线程模块
    thread1 = threading.Thread(target=detection_face)
    thread1.start() # 启动线程

    # 启动主循环
    root.mainloop()

