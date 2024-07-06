import tkinter as tk
from tkinter import font as tkFont
import util
import cv2
from PIL import Image, ImageTk
import pickle
import face_recognition
import os
import datetime
from collections import defaultdict
from Silent_Face_Anti_Spoofing.test import test

class App:
    def __init__(self):
        self.db_dir = './db'
        if not os.path.exists(self.db_dir):
            os.mkdir(self.db_dir)

        self.log_path = './log.txt'
        self.attendance_cooldown = defaultdict(lambda: datetime.datetime(2000, 1, 1))

        self.main_window = tk.Tk()
        self.main_window.geometry("1200x600+200+50")

        # Apply gradient background
        self.create_gradient_background()

        # Custom font
        self.custom_font = tkFont.Font(family="Helvetica", size=14, weight="bold")

        # Title label
        self.title_font = tkFont.Font(family="Helvetica", size=20, weight="bold")
        self.title_label = tk.Label(self.main_window, text="Face Recognition Attendance System",
                                    font=self.title_font, bg='#2E2E2E', fg='white')
        self.title_label.place(x=20, y=20, width=1160, height=50)

        # Webcam frame
        self.webcam_frame = tk.Frame(self.main_window, bg='#FFFFFF', bd=10, relief="ridge")
        self.webcam_frame.place(x=20, y=90, width=740, height=480)

        self.webcam_label = tk.Label(self.webcam_frame, bg='#FFFFFF')
        self.webcam_label.pack(fill="both", expand=True)
        self.add_webcam(self.webcam_label)

        
        self.register_new_user_button_main_window = tk.Button(self.main_window, text="Register New User",
                                                              font=self.custom_font, bg='#4CAF50', fg='white',
                                                              command=self.register_new_user, cursor="hand2",
                                                              bd=0, relief='flat')
        self.register_new_user_button_main_window.place(x=800, y=400, width=300, height=50)

    def create_gradient_background(self):
       
        self.canvas = tk.Canvas(self.main_window, width=1200, height=600)
        self.canvas.pack(fill="both", expand=True)
        
        
        for i in range(1200):
            r = int(255 - (255 - 46) * (i / 1200))
            g = int(255 - (255 - 46) * (i / 1200))
            b = int(255 - (255 - 178) * (i / 1200))
            color = f"#{r:02x}{g:02x}{b:02x}"
            self.canvas.create_line(i, 0, i, 600, fill=color)

        
        self.canvas.pack()

    def add_webcam(self, label):
        if 'cap' not in self.__dict__:
            self.cap = cv2.VideoCapture(0)

        self._label = label
        self.process_webcam()

    def start(self):
        self.main_window.mainloop()

    def process_webcam(self):
        ret, frame = self.cap.read()

        self.most_recent_capture_arr = frame
        img_ = cv2.cvtColor(self.most_recent_capture_arr, cv2.COLOR_BGR2RGB)
        self.most_recent_capture_pil = Image.fromarray(img_)
        imgtk = ImageTk.PhotoImage(image=self.most_recent_capture_pil)
        self._label.imgtk = imgtk
        self._label.configure(image=imgtk)

        self.check_attendance()
        self._label.after(20, self.process_webcam)

    def check_attendance(self):
        anti_spoof_label = test(
            image=self.most_recent_capture_arr,
            model_dir='Silent_Face_Anti_Spoofing//resources//anti_spoof_models',
            device_id=0
        )
        if anti_spoof_label != 1:
            # Anti-spoofing failed, ignore this frame
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

                
                self.show_popup(name)

    def show_popup(self, name):
        popup = tk.Toplevel(self.main_window)
        popup.geometry("200x100+600+300")
        popup.config(bg="#2E2E2E")
        popup_label = tk.Label(popup, text=f" {name} logged In Successfully.", fg="white", bg="#2E2E2E")
        popup_label.pack(expand=True)
        
        popup.after(2000, popup.destroy)


    def register_new_user(self):
        self.register_new_user_window = tk.Toplevel(self.main_window)
        self.register_new_user_window.geometry("800x600+300+100")
        self.register_new_user_window.config(bg="#2E2E2E")

        self.accept_button_register_new_user_window = tk.Button(self.register_new_user_window, text='Accept',
                                                                font=self.custom_font, bg='green', fg='white',
                                                                command=self.accept_register_new_user, cursor="hand2",
                                                                bd=0, relief='flat')
        self.accept_button_register_new_user_window.place(x=550, y=400, width=200, height=50)

        self.try_again_button_register_new_user_window = tk.Button(self.register_new_user_window, text='Try again',
                                                                   font=self.custom_font, bg='red', fg='white',
                                                                   command=self.try_again_register_new_user, cursor="hand2",
                                                                   bd=0, relief='flat')
        self.try_again_button_register_new_user_window.place(x=550, y=480, width=200, height=50)

        self.capture_label = tk.Label(self.register_new_user_window, bg='#FFFFFF', bd=10, relief="ridge")
        self.capture_label.place(x=20, y=20, width=500, height=500)

        self.add_img_to_label(self.capture_label)

        self.entry_text_register_new_user = tk.Text(self.register_new_user_window, height=1, font=self.custom_font, bd=2)
        self.entry_text_register_new_user.place(x=550, y=150, width=200, height=50)

        self.text_label_register_new_user = tk.Label(self.register_new_user_window, text='Please, input username:',
                                                     font=self.custom_font, fg="white", bg="#2E2E2E")
        self.text_label_register_new_user.place(x=550, y=100)

    def try_again_register_new_user(self):
        self.register_new_user_window.destroy()

    def add_img_to_label(self, label):
        imgtk = ImageTk.PhotoImage(image=self.most_recent_capture_pil)
        label.imgtk = imgtk
        label.configure(image=imgtk)

        self.register_new_user_capture = self.most_recent_capture_arr.copy()

    def accept_register_new_user(self):
        name = self.entry_text_register_new_user.get(1.0, "end-1c")
        cleaned_name = name.replace('\n', '').strip()

        face_encodings = face_recognition.face_encodings(self.register_new_user_capture)
        if face_encodings:
            embeddings = face_encodings[0]

            file_path = os.path.join(self.db_dir, f'{cleaned_name}.pickle')
            with open(file_path, 'wb') as file:
                pickle.dump(embeddings, file)

            util.msg_box('Success!', 'User was registered successfully!')
            self.register_new_user_window.destroy()
        else:
            util.msg_box('Error', 'No face detected. Please try again.')

if __name__ == "__main__":
    app = App()
    app.start()
