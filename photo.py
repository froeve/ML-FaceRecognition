import cv2
import os
import time  # 导入time模版
#制定保存图片的文件夹
folder_path = '你的人脸数据库'
# #如果文件夹不存在，则创建它
if not os.path.exists(folder_path):
    os.makedirs(folder_path)
# 调节摄像头拍摄照片
def paizhao():
    # 捕获摄像头
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("无法打开摄像头")
        return

    # 读取并保存图片
    for i in range(1,11):
        ret, frame = cap.read()
        if not ret:
            print(f"无法读取帧{i}")
            continue
        # 构建完整的文件路径
        file_path = os.path.join(folder_path,'yao'+str(i) +'.jpg')
        cv2.imwrite(file_path, frame)
        print(f"已保存图片:{file_path}") # 可选：打印已保存的图片路径

        # 等待3秒
        time.sleep(3)

    # 关闭摄像头
    cap.release()

paizhao()