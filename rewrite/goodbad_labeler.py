import tkinter as tk
from PIL import ImageTk, Image
from dsets.Webcams import Webcams_cls
import os

LABELED_DSET_PATH = '/home/feet/Documents/LAWN/datasets/quality-labeled-webcams/'

def labelImage(imgPath, label, imgIterator, gB, uB, bB, panel):
    cntningDir = os.path.basename(os.path.dirname(imgPath))
    newDir = os.path.join(LABELED_DSET_PATH, label, cntningDir)
    if not os.path.isdir(newDir):
        os.makedirs(newDir)
    
    os.rename(imgPath, os.path.join(newDir, os.path.basename(imgPath)))
    
    try:
        newImgPath = next(imgIterator)
    except:
        print("No more images!")
    
    img = ImageTk.PhotoImage(Image.open(newImgPath).resize((952,718)))
    panel.configure(image=img)
    panel.image = img
    
    gB.configure(command=lambda:labelImage(newImgPath, 'good', imgIterator, gB, uB, bB, panel))
    uB.configure(command=lambda:labelImage(newImgPath, 'unsure', imgIterator, gB, uB, bB, panel))
    bB.configure(command=lambda:labelImage(newImgPath, 'bad', imgIterator, gB, uB, bB, panel))
    

#Make it go through images randomly with an even amount from each class

dset = Webcams_cls('/home/feet/Documents/LAWN/datasets/Webcams/', limits={1.0:300, 1.25:300, 1.5:300, 1.75:300, 2.0:300, 2.25:300, 2.5:300, 3.0:300, 4.0:300, 5.0:300, 6.0:300, 7.0:300, 8.0:300, 9.0:300, 10.0:300})
files = dset.files
imgIter = iter(files)

m = tk.Tk()

imgPath = next(imgIter)
img = ImageTk.PhotoImage(Image.open(imgPath).resize((952,718)))
imgPanel = tk.Label(m, image=img)

buttonsFrame = tk.Frame(m)
buttonsFrame.pack()
goodButton = tk.Button(buttonsFrame, text='Good')
goodButton.grid(row=0, column=0)
unsureButton = tk.Button(buttonsFrame, text='Unsure')
unsureButton.grid(row=0, column=1)
badButton = tk.Button(buttonsFrame, text='Bad')
badButton.grid(row=0, column=2)

imgPanel.pack()

goodButton.configure(command=lambda:labelImage(imgPath, 'good', imgIter, goodButton, unsureButton, badButton, imgPanel))
unsureButton.configure(command=lambda:labelImage(imgPath, 'unsure', imgIter, goodButton, unsureButton, badButton, imgPanel))
badButton.configure(command=lambda:labelImage(imgPath, 'bad', imgIter, goodButton, unsureButton, badButton, imgPanel))

m.mainloop()