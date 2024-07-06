import tkinter as tk
import util
import cv2
from PIL import Image, ImageTk
import pickle
import face_recognition
import os
import datetime
from collections import defaultdict
from Silent_Face_Anti_Spoofing.test import test
import threading

class App:

    def __init__(self):

        self.db_dir = './db'

        if not os.path.exists(self.db_dir):
            os.mkdir(self.db_dir)

        self.log_path = './log.txt'

        self.attendance_cooldown = defaultdict(lambda: datetime.datetime(2000, 1, 1))
        self.attendance_cooldown1 = defaultdict(lambda: datetime.datetime(2000, 1, 1))
        self.main_window = tk.Tk()

        self.main_window.geometry("1200x520+350+100")

        self.register_new_user_button_main_window = util.get_button(self.main_window, 'register new user', 'gray',
                                                                    self.register_new_user, fg='black')

        self.register_new_user_button_main_window.place(x=750, y=400)

        self.webcam_label = util.get_img_label(self.main_window)
        self.webcam_label.place(x=10, y=5, width=400, height=250)

        self.webcam_label1 = util.get_img_label(self.main_window)
        self.webcam_label1.place(x=10, y=260, width=400, height=250)

        self.start_camera_threads()

    def start(self):
        self.main_window.mainloop()

    def start_camera_threads(self):
        cap = cv2.VideoCapture(0)
        cap_new = cv2.VideoCapture(1)

        thread1 = threading.Thread(target=self.process_webcam, args=(self.webcam_label, cap, self.check_attendance))
        thread1.daemon = True
        thread1.start()

        thread2 = threading.Thread(target=self.process_webcam1, args=(self.webcam_label1, cap_new, self.check_attendance1))
        thread2.daemon = True
        thread2.start()

    def process_webcam(self, label, cap, check_attendance_func):
        def update():
            ret, frame = cap.read()
            if ret:
                resized_frame = cv2.resize(frame, (400, 300), fx=0.25, fy=0.25)
                self.most_recent_capture_arr = resized_frame

                img_ = cv2.cvtColor(self.most_recent_capture_arr, cv2.COLOR_BGR2RGB)
                self.most_recent_capture_pil = Image.fromarray(img_)
                imgtk = ImageTk.PhotoImage(image=self.most_recent_capture_pil)
                label.imgtk = imgtk
                label.configure(image=imgtk)

                check_attendance_func()

            label.after(20, update)

        update()

    def process_webcam1(self, label1, cap_new, check_attendance_func):
        def update1():
            ret, frame = cap_new.read()
            if ret:
                resized_frame = cv2.resize(frame, (400, 300))
                self.most_recent_capture_arr1 = resized_frame

                img_ = cv2.cvtColor(self.most_recent_capture_arr1, cv2.COLOR_BGR2RGB)
                self.most_recent_capture_pil1 = Image.fromarray(img_)
                imgtk = ImageTk.PhotoImage(image=self.most_recent_capture_pil1)
                label1.imgtk = imgtk
                label1.configure(image=imgtk)

                check_attendance_func()

            label1.after(20, update1)

        update1()

    def check_attendance(self):
        anti_spoof_label = test(
            image=self.most_recent_capture_arr,
            model_dir='Silent_Face_Anti_Spoofing//resources//anti_spoof_models',
            device_id=0
        )
        if anti_spoof_label != 1:
            return
        name = util.recognize(self.most_recent_capture_arr, self.db_dir)
        if name not in ['unknown_person', 'no_persons_found']:
            cooldown_time = self.attendance_cooldown[name]
            current_time = datetime.datetime.now()
            if (current_time - cooldown_time).total_seconds() >= 30:
                with open(self.log_path, 'a') as f:
                    f.write('{},{},in\n'.format(name, current_time))
                    f.close()
                self.attendance_cooldown[name] = current_time

    def check_attendance1(self):
        anti_spoof_label1 = test(
            image=self.most_recent_capture_arr1,
            model_dir='Silent_Face_Anti_Spoofing//resources//anti_spoof_models',
            device_id=0
        )

        if anti_spoof_label1 != 1:
            return

        name = util.recognize(self.most_recent_capture_arr1, self.db_dir)

        if name not in ['unknown_person', 'no_persons_found']:
            cooldown_time = self.attendance_cooldown1[name]
            current_time = datetime.datetime.now()

            if (current_time - cooldown_time).total_seconds() >= 30:
                with open(self.log_path, 'a') as f:
                    f.write('{},{},out\n'.format(name, current_time))
                    f.close()

                self.attendance_cooldown1[name] = current_time

    def register_new_user(self):
        self.register_new_user_window = tk.Toplevel(self.main_window)
        self.register_new_user_window.geometry("1200x520+370+120")
        self.accept_button_register_new_user_window = util.get_button(self.register_new_user_window, 'Accept', 'green',
                                                                      self.accept_register_new_user)
        self.accept_button_register_new_user_window.place(x=750, y=300)
        self.try_again_button_register_new_user_window = util.get_button(self.register_new_user_window, 'Try again',
                                                                         'red', self.try_again_register_new_user)
        self.try_again_button_register_new_user_window.place(x=750, y=400)
        self.capture_label = util.get_img_label(self.register_new_user_window)
        self.capture_label.place(x=10, y=0, width=700, height=500)
        self.add_img_to_label(self.capture_label)
        self.entry_text_register_new_user = util.get_entry_text(self.register_new_user_window)
        self.entry_text_register_new_user.place(x=750, y=150)
        self.text_label_register_new_user = util.get_text_label(self.register_new_user_window,
                                                                'Please, \ninput username:')
        self.text_label_register_new_user.place(x=750, y=70)

    def try_again_register_new_user(self):
        self.register_new_user_window.destroy()

    def add_img_to_label(self, label):
        imgtk = ImageTk.PhotoImage(image=self.most_recent_capture_pil)
        label.imgtk = imgtk
        label.configure(image=imgtk)
        self.register_new_user_capture = self.most_recent_capture_arr.copy()

    def add_img_to_label1(self, label):
        try:
            imgtk = ImageTk.PhotoImage(image=self.most_recent_capture_pil1)
            label.imgtk = imgtk
            label.configure(image=imgtk)
            self.register_new_user_capture1 = self.most_recent_capture_arr1.copy()

        except Exception as e:
            print(f"Error in add_img_to_label1: {e}")

    def accept_register_new_user(self):
        name = self.entry_text_register_new_user.get(1.0, "end-1c")
        cleaned_name = name.replace('\n', '').strip()
        embeddings = face_recognition.face_encodings(self.register_new_user_capture)[0]
        file = open(os.path.join(self.db_dir, '{}.pickle'.format(cleaned_name)), 'wb')
        pickle.dump(embeddings, file)
        util.msg_box('Success!', 'User was registered successfully!')
        self.register_new_user_window.destroy()

if __name__ == "__main__":
    app = App()
    app.start()
