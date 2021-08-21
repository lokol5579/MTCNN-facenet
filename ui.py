import tkinter as tk
import cv2
from tkinter import filedialog
from face_recognize import face_rec

def start_camera():
    # 生成人脸识别类，并初始化
    my_face_rec = face_rec()

    # 打开摄像头
    video = cv2.VideoCapture(0)

    while True:
        temp, draw = video.read()
        my_face_rec.recognize(draw)
        cv2.imshow('Video', draw)
        if cv2.waitKey(20) & 0xFF == ord('q'):
            break

    video.release()
    cv2.destroyAllWindows()

def choose_pic():
    path = filedialog.askopenfilename()

    # 生成人脸识别类，并初始化
    my_face_rec = face_rec()

    while True:
        pic = cv2.imread(path)
        my_face_rec.recognize(pic)
        cv2.imshow('Picture', pic)
        if cv2.waitKey(20) & 0xFF == ord('q'):
            break

    cv2.destroyAllWindows()

def choose_video():
    path = filedialog.askopenfilename()

    # 生成人脸识别类，并初始化
    my_face_rec = face_rec()

    # 打开摄像头
    video = cv2.VideoCapture(path)

    while True:
        temp, draw = video.read()
        my_face_rec.recognize(draw)
        cv2.imshow('Video', draw)
        if cv2.waitKey(20) & 0xFF == ord('q'):
            break

    video.release()
    cv2.destroyAllWindows()

def creat_window():
    window = tk.Tk()
    window.geometry('460x120')
    window.title('人脸及表情识别')

    btn1 = tk.Button(window, text='开启摄像头', command=lambda:[window.destroy(), start_camera()])
    btn1.place(relx=0.1, rely=0.4, relwidth=0.2, relheight=0.2)
    btn2 = tk.Button(window, text='选择图片', command=lambda:[window.destroy(),choose_pic])
    btn2.place(relx=0.4, rely=0.4, relwidth=0.2, relheight=0.2)
    btn3 = tk.Button(window, text='选择视频', command=lambda:[window.destroy(),choose_video])
    btn3.place(relx=0.7, rely=0.4, relwidth=0.2, relheight=0.2)

    window.mainloop()

if __name__ == "__main__":
    creat_window()