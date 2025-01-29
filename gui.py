import tkinter as tk
from PIL import ImageTk, Image
from tkinter import filedialog
import numpy as np
import tensorflow
import keras
from keras.preprocessing import image
from keras.preprocessing.image import img_to_array
from keras.models import load_model



def load_img():
    global img, image_data
    for img_display in frame.winfo_children():
        img_display.destroy()

    image_data = filedialog.askopenfilename(initialdir="/", title="Choose an image",
                                       filetypes=(("all files", "*.*"), ("png files", "*.png")))
    img = Image.open(image_data)
    # wpercent = (basewidth / float(img.size[0]))
    img = img.resize((226,226), Image.ANTIALIAS)
    img = ImageTk.PhotoImage(img)
    file_name = image_data.split('/')
    panel = tk.Label(frame, text= str(file_name[len(file_name)-1]).upper()).pack()
    panel_image = tk.Label(frame, image=img).pack()


def classify():
    original = Image.open(image_data)
    original = original.resize((226, 226), Image.ANTIALIAS)
    numpy_image = img_to_array(original)
    image_batch = np.expand_dims(numpy_image, axis=0)
    print('PREDICT HERE')

    model = load_model('Trained_model.h5')

    pred = model.predict( numpy_image )
    print(np.round(pred))

    table = tk.Label(frame, text= str(np.round(pred)) , fg = 'black').pack()
    print('HERE')

    # table = tk.Label(frame, text="Top image class predictions and confidences").pack()
    # for i in range(0, len(label[0])):
    #      result = tk.Label(frame,
    #                 text= str(label[0][i][1]).upper() + ': ' +
    #                        str(round(float(label[0][i][2])*100, 3)) + '%').pack()

root = tk.Tk()
root.title('Portable Image Classifier')
# root.iconbitmap('class.ico')
root.resizable(False, False)
tit = tk.Label(root, text="Portable Image Classifier", padx=25, pady=6, font=("", 12)).pack()
canvas = tk.Canvas(root, height=500, width=500, bg='grey')
canvas.pack()
frame = tk.Frame(root, bg='white')
frame.place(relwidth=0.8, relheight=0.8, relx=0.1, rely=0.1)
chose_image = tk.Button(root, text='Choose Image',
                        padx=35, pady=10,
                        fg="white", bg="grey", command=load_img)
chose_image.pack(side=tk.LEFT)
class_image = tk.Button(root, text='Classify Image',
                        padx=35, pady=10,
                        fg="white", bg="grey", command=classify)
class_image.pack(side=tk.RIGHT)

root.mainloop()





