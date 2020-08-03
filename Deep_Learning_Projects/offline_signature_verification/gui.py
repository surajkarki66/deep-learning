import tkinter as tk
import numpy as np
import cv2

from tkinter import *
from tkinter import filedialog
from PIL import Image, ImageTk
from model import SigNet
from utils import CosineSimilarity


class MainWindow(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Signature Verifier")
        self.geometry("700x600")
        self.resizable(False, False)
        self.configure(background='#CDCDCD')

        # Define labels

        self.upload = Button(self, text="Upload an image", command=self.upload_image,
                             padx=10, pady=5)

        self.upload.configure(background='#364156', foreground='white',
                              font=('arial', 10, 'bold'))
        self.upload.pack(side=BOTTOM, pady=50)

        self.signature_image = Label(self)
        self.definition = Label(
            self, background='#CDCDCD', font=('arial', 20, 'bold'))
        self.heading = Label(self, text="Offline Signature Verifier",
                             pady=20, font=('arial', 30, 'bold'))
        self.heading.configure(background='#CDCDCD', foreground='#364156')
        self.heading.pack(side=TOP, expand=True)
        self.signature_image.pack(side=BOTTOM, expand=True)
        self.definition.pack(side=BOTTOM)

        self.model = SigNet()

    def database_image(self, file_path):
        img = cv2.imread(file_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img = cv2.resize(img, (300, 150))
        retval, img = cv2.threshold(img, 0, 255, type=cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
        img = img / 255.
        img = img.reshape(1, 300, 150, 1)
        emb = self.model.predict([img])[0]
        return emb

    def get_database_emb(self):
        emb = []
        for i in range(5):
            file_path = f'./database/{i+1}.jpg'
            pred = self.database_image(file_path)
            pred = pred.reshape(128,)
            emb.append(pred)

        return emb


    def verify(self, file_path):
        img = cv2.imread(file_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img = cv2.resize(img, (300, 150))
        retval, img = cv2.threshold(img, 0, 255, type=cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
        img = img / 255.
        img = img.reshape(1, 300, 150, 1)
        emb1 = self.get_database_emb()
        emb2 = self.model.predict([img])[0]
        emb2 = emb2.reshape(128,)
        similarities = []
        for i in emb1:
            cosine_similarity = CosineSimilarity(i, emb2)
            similarities.append(cosine_similarity)
        similarities = np.array(similarities)
        max_similarity = np.amax(similarities)
      
        if max_similarity > 0.85:
            text = "Congratulation! your signature is Geniune."
        else:
            text = "Oops! your signature is Forged."

        self.definition.configure(foreground='#011638', text=text)
        



    def show_verify_button(self, file_path):
        verify = Button(self, text="Verify Signature",
                        command=lambda: self.verify(file_path), padx=10, pady=5)
        verify.configure(background='#364156', foreground='white',
                         font=('arial', 10, 'bold'))
        verify.place(relx=0.79, rely=0.46)

    def upload_image(self):
        try:
            file_path = filedialog.askopenfilename()
            uploaded = Image.open(file_path)
            uploaded.thumbnail(
                ((self.winfo_width()/2.25), (self.winfo_height()/2.25)))
            im = ImageTk.PhotoImage(uploaded)
            self.signature_image.configure(image=im)
            self.signature_image.image = im
            self.show_verify_button(file_path)
            self.definition.configure(foreground='#011638', text="")
        

        except:
            pass
