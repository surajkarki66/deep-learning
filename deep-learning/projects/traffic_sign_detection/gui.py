import tkinter as tk
import pandas as pd
import pickle
import numpy as np

from tkinter import filedialog
from tkinter import *
from PIL import ImageTk, Image
from keras.models import load_model


class MainWindow(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("TRAFFIC SIGN CLASSIFIER")
        self.geometry("800x600")
        self.resizable(False, False)
        self.configure(background='#CDCDCD')

        # Defininf labels
        self.class_name = Label(
            self, background='#CDCDCD', font=('arial', 20, 'bold'))
        self.accuracy_label = Label(self, font=('arial', 20, 'bold'))
        self.class_image = Label(self)
        self.definition = Label(
            self, background='#CDCDCD', font=('arial', 20, 'bold'))

        self.upload = Button(self, text="Upload an image", command=self.upload_image,
                             padx=10, pady=5)

        self.upload.configure(background='#364156', foreground='white',
                              font=('arial', 10, 'bold'))
        self.upload.pack(side=BOTTOM, pady=50)
        self.definition.pack(side=BOTTOM)
        self.class_image.pack(side=BOTTOM, expand=True)
        self.accuracy_label.pack(side=BOTTOM, expand=True)
        self.class_name.pack(side=BOTTOM, expand=True)
        self.heading = Label(self, text="TRAFFIC SIGN CLASSIFIER",
                             pady=20, font=('arial', 30, 'bold'))
        self.heading.configure(background='#CDCDCD', foreground='#364156')
        self.heading.pack(side=TOP, expand=True)

        pickle_in = open("traffic_model.p", "rb")  # rb = READ BYTE
        self.model = pickle.load(pickle_in)
        # model.summary()

    def get_classname(self, class_no):
        data = pd.read_csv('labels.csv')
        classname = None
        for class_id, class_name in data.values:
            if class_no == class_id:
                classname = class_name

        return classname

    def classify(self, file_path):
        image = Image.open(file_path)
        image = image.resize((32, 32))
        image = image.convert('1')
        image = np.expand_dims(image, axis=0)
        image = np.array(image)
        image = image.reshape((1, 32, 32, 1))
        pred = self.model.predict([image])[0]
        accuracy = np.amax(pred) * 100
        accuracy = 'Accuracy:' + str(accuracy) + '%'
        self.accuracy_label.configure(foreground='#011638', text=accuracy)
        pred = np.argmax(pred)
        class_name = self.get_classname(pred)
        self.class_name.configure(foreground='#011638', text=class_name)
        text = 'This is a ' + str(class_name)
        self.definition.configure(foreground='#011638', text=text)

    def show_classify_button(self, file_path):
        classify_b = Button(self, text="Classify Image",
                            command=lambda: self.classify(file_path), padx=10, pady=5)
        classify_b.configure(background='#364156', foreground='white',
                             font=('arial', 10, 'bold'))
        classify_b.place(relx=0.79, rely=0.46)

    def upload_image(self):
        try:
            file_path = filedialog.askopenfilename()
            uploaded = Image.open(file_path)
            uploaded.thumbnail(
                ((self.winfo_width()/2.25), (self.winfo_height()/2.25)))
            im = ImageTk.PhotoImage(uploaded)
            self.class_image.configure(image=im)
            self.class_image.image = im
            self.class_name.configure(text='')
            self.definition.configure(text='')
            self.accuracy_label.configure(text='')
            self.show_classify_button(file_path)
        except:
            pass

