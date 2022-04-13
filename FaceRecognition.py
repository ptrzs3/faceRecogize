import os
import cv2
import numpy as np
import dlib
import sys

def face_recognize(face_vec, face_dataset, ids):
    N = face_dataset.shape[0]
    diffMat = np.tile(face_vec, (N, 1)) - face_dataset

    # 计算欧式距离
    distances = np.linalg.norm(diffMat, axis=1)

    # 找到最小距离
    idx = np.argmin(distances)

    # 返回id编号与距离
    return ids[idx], distances[idx]


# 返回-1，缺少配置文件
# 返回-2，缺少模型文件
# 返回-2，未识别用户
# 返回字符串，识别用户
class FaceRecognition:
    def __init__(self):
        # 加载Opencv人脸检测器
        self.dir_name = os.path.dirname(os.path.realpath(sys.argv[0]))
        self.faceCascade_file = os.path.join(self.dir_name, 'haarcascade_frontalface_alt.xml')
        self.facerec_file = os.path.join(self.dir_name, 'dlib_face_recognition_resnet_model_v1.dat')
        self.sp_file = os.path.join(self.dir_name, 'shape_predictor_5_face_landmarks.dat')
        self.check_file_exist()

        self.faceCascade = cv2.CascadeClassifier(self.faceCascade_file)
        # 加载人脸特征提取器
        self.facerec = dlib.face_recognition_model_v1(os.path.join(self.facerec_file))
        # 加载人脸标志点检测器
        self.sp = dlib.shape_predictor(os.path.join(self.sp_file))
        # Dlib 人脸检测器
        self.detector = dlib.get_frontal_face_detector()

    def check_file_exist(self):
        if not os.path.exists(self.sp_file) & os.path.exists(self.facerec_file) & os.path.exists(self.faceCascade_file):
            print(-1)
            sys.exit(0)

    def recognize(self):
        # 加载训练好的人脸模型
        try:
            model = np.load(os.path.join(self.dir_name, 'trainer.npz'))
        except FileNotFoundError:
            print(-2)
            sys.exit(0)

        face_vectors = model['face_vectors']
        face_ids = model['ids']

        # 打开摄像头
        cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
        flag = True
        while flag:
            # 读取一帧图像
            success, img = cap.read()

            if not success:
                continue

            # BGR 转 gray
            img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            # BGR 转 RGB
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            # 进行人脸检测
            dets = self.detector(img_gray, 1)

            # 遍历检测的人脸
            for k, d in enumerate(dets):
                # 画框
                cv2.rectangle(img, (d.left(), d.top()), (d.right(), d.bottom()), (255, 0, 0), 3)

                # 标志点检测
                shape = self.sp(img_rgb, d)

                # 获取人脸特征
                face_vector = self.facerec.compute_face_descriptor(img_rgb, shape)

                # 进行识别返回ID与距离
                face_id, dis = face_recognize(np.array(face_vector), face_vectors, face_ids)

                if dis < 0.38:
                    print(face_id)
                    flag = False
                else:
                    print(-3)
                    flag = False
            # 显示检测结果
            cv2.imshow("FACE", img)

            # 按键 "q" 退出
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()


if __name__ == '__main__':
    pwd = sys.argv[1]
    if pwd != 'aE1pG0aB1gE4':
        sys.exit(0)
    fd = FaceRecognition()
    fd.recognize()