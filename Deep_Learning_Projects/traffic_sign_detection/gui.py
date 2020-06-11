import tkinter as tk
import pandas as pd
import pickle

from tkinter import filedialog
from tkinter import *
from PIL import ImageTk, Image
from tensorflow.keras.models import load_model

pickle_in=open("traffic_model.p","rb")  ## rb = READ BYTE
model=pickle.load(pickle_in)

def get_classname(class_no):
    data = pd.read_csv('labels.csv')
    classname= None 
    for class_id, class_name in data.values:
        if class_no == class_id:
            classname = class_name

    return classname

def classify(file_path):
    image = Image.open(file_path)
    image = image.resize((32, 32))
    image = np.expand_dims(image, axis=0)
    image = np.array(image)
    pred = model.predict([image])[0]
    pred = np.argmax(pred)
    sign = get_classname(pred)
    label.configure(foreground='#011638', text=sign)


def show_classify_button(file_path):
    classify_b = Button(top, text="Classify Image",
                        command=lambda: classify(file_path), padx=10, pady=5)
    classify_b.configure(background='#364156', foreground='white',
                         font=('arial', 10, 'bold'))
    classify_b.place(relx=0.79, rely=0.46)


def upload_image():
    try:
        file_path = filedialog.askopenfilename()
        uploaded = Image.open(file_path)
        uploaded.thumbnail(
            ((top.winfo_width()/2.25), (top.winfo_height()/2.25)))
        im = ImageTk.PhotoImage(uploaded)
        sign_image.configure(image=im)
        sign_image.image = im
        label.configure(text='')
        show_classify_button(file_path)
    except:
        pass


if __name__ == "__main__":
   
    # dictionary to label all the CIFAR-10 dataset classes.

    # initialise GUI
    top = tk.Tk()
    top.geometry('800x600')
    top.title('Image Classification CIFAR10')
    top.configure(background='#CDCDCD')
    label = Label(top, background='#CDCDCD', font=('arial', 15, 'bold'))
    sign_image = Label(top)

    upload = Button(top, text="Upload an image", command=upload_image,
                    padx=10, pady=5)
    upload.configure(background='#364156', foreground='white',
                     font=('arial', 10, 'bold'))
    upload.pack(side=BOTTOM, pady=50)
    sign_image.pack(side=BOTTOM, expand=True)
    label.pack(side=BOTTOM, expand=True)
    heading = Label(top, text="Image Classification CIFAR10",
                    pady=20, font=('arial', 20, 'bold'))
    heading.configure(background='#CDCDCD', foreground='#364156')
    heading.pack()
    top.mainloop()
