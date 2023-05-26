import cv2
import dlib
import numpy as np
import pygame
from imutils import face_utils
import tkinter as tk
from tkinter import ttk

from threading import Thread, Event

def eye_aspect_ratio(eye):
    A = np.linalg.norm(eye[1] - eye[5])
    B = np.linalg.norm(eye[2] - eye[4])
    C = np.linalg.norm(eye[0] - eye[3])

    ear = (A + B) / (2.0 * C)
    return ear

def play_song(song_file):
    pygame.mixer.init()
    pygame.mixer.music.load(song_file)
    pygame.mixer.music.play()

def track_face_and_eyes(start_event, stop_event):
    pygame.mixer.init()
    cap = cv2.VideoCapture(0)
    predictor_path = "shape_predictor_68_face_landmarks.dat"
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor(predictor_path)

    music_playing = False

    while not stop_event.is_set():
        if not start_event.is_set():
            continue

        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        rects = detector(gray, 0)

        for rect in rects:
            shape = predictor(gray, rect)
            shape = face_utils.shape_to_np(shape)

            left_eye = shape[42:48]
            right_eye = shape[36:42]

            left_eye_aspect_ratio = eye_aspect_ratio(left_eye)
            right_eye_aspect_ratio = eye_aspect_ratio(right_eye)

            average_eye_aspect_ratio = (left_eye_aspect_ratio + right_eye_aspect_ratio) / 2.0

            if average_eye_aspect_ratio < 0.25:  # Gözler kapalıysa
                if not music_playing:
                    play_song("demet.mp3")
                    music_playing = True
            else:
                if music_playing:
                    pygame.mixer.music.stop()
                    music_playing = False

    cap.release()
    cv2.destroyAllWindows()

def main():
    start_event = Event()
    stop_event = Event()

    def on_start_button_click():
        start_event.set()

    def on_stop_button_click():
        start_event.clear()

    def on_close():
        stop_event.set()
        root.destroy()

    root = tk.Tk()
    root.title("Dikkat Takip Uygulaması")
    # Pencere boyutunu ayarlama
    root.geometry("600x500")
    root.configure(bg="lightblue")






    start_button = tk.Button(root, text="Başlat", command=on_start_button_click,width=20, height=6)
    start_button.pack()
    # Başlat düğmesinin boyut ve renk ayarları
    start_button.config(width=60, background="green", foreground="white")
    start_button.pack(pady=(100, 10), padx=50)

    stop_button = tk.Button(root, text="Durdur", command=on_stop_button_click,width=20, height=6)
    stop_button.pack()
    # Durdur düğmesinin boyut ve renk ayarları
    stop_button.config(width=60, background="red", foreground="white")
    hello_label = tk.Label(root, text="İbrahim Serhat Demircioğlu", font=("Helvetica", 16))
    hello_label.pack(pady=10)


    hello_label = tk.Label(root, text="Bitirme Projesi", font=("Helvetica", 16))
    hello_label.pack(pady=10)

    root.protocol("WM_DELETE_WINDOW", on_close)

    tracking_thread = Thread(target=track_face_and_eyes, args=(start_event, stop_event))
    tracking_thread.start()

    root.mainloop()

if __name__ == "__main__":
    main()