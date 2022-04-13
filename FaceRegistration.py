import sys

import cv2
import os
import dlib
import time
import shutil
import numpy as np

dir_name = os.path.dirname(os.path.realpath(sys.argv[0]))
faceCascade_file = os.path.join(dir_name, 'haarcascade_frontalface_alt.xml')
facerec_file = os.path.join(dir_name, 'dlib_face_recognition_resnet_model_v1.dat')
sp_file = os.path.join(dir_name, 'shape_predictor_5_face_landmarks.dat')
model_file = os.path.join(dir_name, 'trainer.npz')


def get_face_list(path):
    for root, dirs, _ in os.walk(path):
        if root == path:
            return dirs


# 返回-1，缺少探测器文件
# 返回0，正常退出
def face_register(name):
    if not os.path.exists(sp_file) & os.path.exists(facerec_file) & os.path.exists(faceCascade_file):
        print(-1)
        sys.exit(0)
    faceCascade = cv2.CascadeClassifier(faceCascade_file)
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    start_time = time.time()
    end_time = time.time()
    index_photo = 1
    save_dir_p = os.path.join(dir_name, "user_face")
    save_dir_s = os.path.join(save_dir_p, "unnamed_user")
    if not os.path.exists(save_dir_s):
        os.makedirs(save_dir_s)
    while True:
        success, img = cap.read()
        if not success:
            continue

        # 转换为灰度
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # 进行人脸检测
        faces = faceCascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(50, 50),
                                                  flags=cv2.CASCADE_SCALE_IMAGE)

        # 画框
        face_detected = False
        for (x, y, w, h) in faces:
            cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 3)
            cv2.putText(img, "face recognized", (x + 5, y - 5), cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 255), 2)
            face_detected = True
        if face_detected and end_time - start_time > 2 and index_photo < 11:
            start_time = time.time()
            roi = img[y:y + h, x:x + w]
            cv2.imwrite("%s/%d.jpg" % (save_dir_s, index_photo), roi)
            index_photo = index_photo + 1
        end_time = time.time()
        # 显示检测结果
        cv2.imshow("shooting", img)
        cv2.waitKey(1)
        if index_photo == 11:
            cv2.destroyWindow("shooting")
            break

    try:
        os.rename(save_dir_s, os.path.join(save_dir_p, name))
    except FileExistsError:
        shutil.rmtree(os.path.join(save_dir_p, name))
        os.rename(save_dir_s, os.path.join(save_dir_p, name))
    train_model()


def train_model():
    # 人脸识别训练图片存放路径
    base_path = os.path.join(dir_name, 'user_face')
    # 加载人脸特征提取器

    facerec = dlib.face_recognition_model_v1(facerec_file)

    # 加载人脸标志点检测器

    sp = dlib.shape_predictor(sp_file)
    # 获取人脸 id 列表
    face_list = get_face_list(base_path)

    # 用来存储人脸特征和人脸id的列表
    list_face_vector = []
    list_face_id = []

    for face_id in face_list:

        for f_img in os.listdir(os.path.join(base_path, face_id)):
            if f_img.endswith(".jpg"):
                file_img = os.path.join(base_path, face_id, f_img)

                # 读取图像并转换为RGB
                img = cv2.imread(file_img)
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

                # 在整图内检测标志点
                img = np.array(img)
                h, w, _ = np.shape(img)
                rect = dlib.rectangle(0, 0, w, h)
                shape = sp(img, rect)

                # 获取128维人脸特征
                face_vector = facerec.compute_face_descriptor(img, shape)

                # 特征 和 id 保存
                list_face_vector.append(face_vector)
                list_face_id.append(face_id)

    # 将最终结果进行保存
    face_vectors = np.array(list_face_vector)
    ids = np.array(list_face_id)

    # 模型保存
    np.savez(model_file, face_vectors=face_vectors, ids=ids)
    print(0)


if __name__ == '__main__':
    username = sys.argv[1]
    face_register(username)