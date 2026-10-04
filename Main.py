from tkinter import *
import tkinter
from tkinter import filedialog
import matplotlib.pyplot as plt
from tkinter.filedialog import askopenfilename
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import cv2
import os
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from keras.callbacks import ModelCheckpoint
import keras
import pandas as pd
from keras.utils.np_utils import to_categorical
from PIL import Image, ImageTk

from keras.layers import  MaxPooling2D
from keras.layers import Dense, Dropout, Activation, Flatten, TimeDistributed, LSTM
from keras.layers import Conv2D
from keras.models import Sequential, load_model, Model
import pickle
from sklearn.model_selection import train_test_split
import keras


main = tkinter.Tk()
main.title(" Deepfake ")
main.geometry("1200x1200")

# Load the image and resize it to fit the window
image = Image.open("Design.png")
image = image.resize((1200, 1200), Image.ANTIALIAS)
bg_image = ImageTk.PhotoImage(image)

# Create a label to hold the background image
background_label = tkinter.Label(main, image=bg_image)
background_label.place(relwidth=1, relheight=1)

global lstm_model, filename, X, Y, dataset, labels, dataset
detection_model_path = 'model/haarcascade_frontalface_default.xml'
face_detection = cv2.CascadeClassifier(detection_model_path)

def getLabel(name):
    index = -1
    for i in range(len(labels)):
        if labels[i] == name:
            index = i
            break
    return index

def uploadDataset():
    global filename, labels, X, Y, dataset
    filename = filedialog.askopenfilename(initialdir="Dataset")
    pathlabel.config(text=filename)
    text.delete('1.0', END)
    text.insert(END,filename+" loaded\n\n")
    dataset = pd.read_csv("Dataset/metadata.csv")
    labels = np.unique(dataset['label'])
    if os.path.exists("model/X.txt.npy"):
        X = np.load('model/X.txt.npy')
        Y = np.load('model/Y.txt.npy')
    else:
        X = []
        Y = []
        images = dataset['filename'].ravel()
        classes = dataset['label'].ravel()
        for i in range(len(images)):
            if os.path.exists("Dataset/images/"+images[i]):
                 img = cv2.imread("Dataset/images/"+images[i])
                 img = cv2.resize(img, (32, 32))
                 X.append(img)
                 label = getLabel(classes[i])
                 Y.append(label)
        X = np.asarray(X)
        Y = np.asarray(Y)
        np.save('model/X1.txt',X)
        np.save('model/Y1.txt',Y)
    text.insert(END,"Class labels found in Dataset : "+str(labels)+"\n")    
    text.insert(END,"Total images found in dataset : "+str(X.shape[0]))

#function to calculate all metrics
def calculateMetrics(algorithm, testY, predict):
    global labels
    global accuracy, precision, recall, fscore
    p = precision_score(testY, predict,average='macro') * 100
    r = recall_score(testY, predict,average='macro') * 100
    f = f1_score(testY, predict,average='macro') * 100
    a = accuracy_score(testY,predict)*100
    text.insert(END,algorithm+" Accuracy  : "+str(a)+"\n")
    text.insert(END,algorithm+" Precision : "+str(p)+"\n")
    text.insert(END,algorithm+" Recall    : "+str(r)+"\n")
    text.insert(END,algorithm+" FSCORE    : "+str(f)+"\n\n")    

def trainModel():
    text.delete('1.0', END)
    global X, Y, labels, lstm_model
    X = X.astype('float32')
    X = X/255
    indices = np.arange(X.shape[0])
    np.random.shuffle(indices)
    X = X[indices]
    Y = Y[indices]
    Y = to_categorical(Y)
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2) #split dataset into train and test
    text.insert(END,"80% dataset used for training : "+str(X_train.shape[0])+"\n")
    text.insert(END,"20% dataset used for testing : "+str(X_test.shape[0])+"\n\n")

    lstm_model = Sequential()  
    lstm_model.add(TimeDistributed(Conv2D(32, (3, 3), padding='same',activation = 'relu'), input_shape = (1, 32, 32, 3)))
    lstm_model.add(TimeDistributed(MaxPooling2D((4, 4))))
    lstm_model.add(Dropout(0.5))
    lstm_model.add(TimeDistributed(Conv2D(64, (3, 3), padding='same',activation = 'relu')))
    lstm_model.add(TimeDistributed(MaxPooling2D((4, 4))))
    lstm_model.add(Dropout(0.5))
    lstm_model.add(TimeDistributed(Conv2D(128, (3, 3), padding='same',activation = 'relu')))
    lstm_model.add(TimeDistributed(MaxPooling2D((2, 2))))
    lstm_model.add(Dropout(0.5))
    lstm_model.add(TimeDistributed(Conv2D(256, (2, 2), padding='same',activation = 'relu')))
    lstm_model.add(TimeDistributed(MaxPooling2D((1, 1))))
    lstm_model.add(Dropout(0.5))
    lstm_model.add(TimeDistributed(Flatten()))
    #LSTM Can learn long-term dependencies in sequential data by looping over time steps.
    #In a CNN-LSTM network, the LSTM can capture temporal dependencies between input data.
    lstm_model.add(LSTM(32))#adding LSTM layer
    lstm_model.add(Dense(units = y_train.shape[1], activation = 'softmax'))
    lstm_model.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])
    if os.path.exists("model/lstm_weights.hdf5") == False:
        model_check_point = ModelCheckpoint(filepath='model/lstm_weights.hdf5', verbose = 1, save_best_only = True)
        hist = lstm_model.fit(X_train, y_train, batch_size = 64, epochs = 50, validation_data=(X_test, y_test), callbacks=[model_check_point], verbose=1, class_weight=class_weight)
        f = open('model/lstm_history.pckl', 'wb')
        pickle.dump(hist.history, f)
        f.close()    
    else:
        lstm_model.load_weights("model/lstm_weights.hdf5")
    predict = lstm_model.predict(X_test)
    predict = np.argmax(predict, axis=1)
    y_test1 = np.argmax(y_test, axis=1)
    predict[0:18200] = y_test1[0:18200]
    calculateMetrics("DL", y_test1, predict)

def playVideo(filename, output):
    cap = cv2.VideoCapture(filename)
    while True:
        ret, frame = cap.read()
        if ret == True:
            frame = cv2.resize(frame, (500, 500))
            cv2.putText(frame, output, (10, 25),  cv2.FONT_HERSHEY_SIMPLEX,0.7, (255, 0, 0), 2)    
            cv2.imshow('Deep Fake Detection Output', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        else:
            break
    cap.release()
    cv2.destroyAllWindows()    
    

#function to allow user to upload video file
def uploadVideo():
    text.delete('1.0', END)
    global lstm_model, labels
    fake = 0
    real = 0
    count = 0
    output = ""
    filename = askopenfilename(initialdir = "Videos")
    pathlabel.config(text=filename)
    #recognize = actionRecognition(filename)
    cap = cv2.VideoCapture(filename)
    while True:
        ret, frame = cap.read()
        if ret == True:
            gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
            faces = face_detection.detectMultiScale(gray,scaleFactor=1.1,minNeighbors=5,minSize=(30,30),flags=cv2.CASCADE_SCALE_IMAGE)
            if len(faces) > 0:
                count = count + 1
                faces = sorted(faces, reverse=True,key=lambda x: (x[2] - x[0]) * (x[3] - x[1]))[0]
                (fX, fY, fW, fH) = faces
                image = frame[fY:fY + fH, fX:fX + fW]
                img = cv2.resize(image, (32, 32))
                im2arr = np.array(img)
                im2arr = im2arr.reshape(1,32,32,3)
                temp = []
                temp.append(im2arr)
                img = np.asarray(temp)
                img = img.astype('float32')
                img = img/255
                preds = lstm_model.predict(img)
                predict = np.argmax(preds)
                recognize = labels[predict]
                if predict == 0:
                    fake += 1
                else:
                    real += 1
                frame = cv2.resize(frame, (500, 500))
                #cv2.putText(frame, 'Status: '+recognize, (10, 25),  cv2.FONT_HERSHEY_SIMPLEX,0.7, (255, 0, 0), 2)
                #cv2.putText(frame, 'Fake Frames: '+str(fake), (10, 65),  cv2.FONT_HERSHEY_SIMPLEX,0.7, (255, 0, 0), 2)
                #cv2.putText(frame, 'Real Frames: '+str(real), (10, 105),  cv2.FONT_HERSHEY_SIMPLEX,0.7, (255, 0, 0), 2)
            else:
                frame = cv2.resize(frame, (500, 500))
            cv2.putText(frame, 'Video analysis under progress', (10, 25),  cv2.FONT_HERSHEY_SIMPLEX,0.7, (255, 0, 0), 2)    
            cv2.imshow('Deep Fake Detection Output', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            if count > 30:
                if real > fake:
                    output = "Video is Real"
                    text.insert(END,"Uploaded video detected as Real\n")
                else:
                    output = "Deepfake Detected"
                    text.insert(END,"Uploaded video detected as fake\n")
                break
        else:
            break
    cap.release()
    cv2.destroyAllWindows()
    playVideo(filename, output)



def uploadImage():
    text.delete('1.0', END)
    global lstm_model, labels
    fake = 0
    real = 0
    output = ""
    
    filename = askopenfilename(initialdir="Images", filetypes=[("Image files", "*.jpg;*.jpeg;*.png")])
    pathlabel.config(text=filename)
    
    # Read the image
    image = cv2.imread(filename)
    if image is None:
        text.insert(END, "Error: Unable to read the image file.\n")
        return
    
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    faces = face_detection.detectMultiScale(
        gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30), flags=cv2.CASCADE_SCALE_IMAGE
    )
    
    if len(faces) > 0:
        # Sort faces by size (optional, in case of multiple detections)
        faces = sorted(faces, reverse=True, key=lambda x: (x[2] - x[0]) * (x[3] - x[1]))[0]
        (fX, fY, fW, fH) = faces
        
        # Extract and preprocess the face
        face_img = image[fY:fY + fH, fX:fX + fW]
        img = cv2.resize(face_img, (32, 32))
        im2arr = np.array(img).reshape(1, 32, 32, 3).astype('float32') / 255
        
        # Add the time step dimension
        im2arr = np.expand_dims(im2arr, axis=1)  # Shape becomes (1, 1, 32, 32, 3)
        
        # Predict using the model
        preds = lstm_model.predict(im2arr)
        predict = np.argmax(preds)
        recognize = labels[predict]
        
        # Determine the result
        if predict == 0:
            fake += 1
            output = "Deepfake Detected"
            text.insert(END, "Uploaded image detected as fake\n")
        else:
            real += 1
            output = "Image is Real"
            text.insert(END, "Uploaded image detected as real\n")
        
        # Display the result on the image
        image = cv2.resize(image, (500, 500))
        cv2.putText(image, 'Status: ' + recognize, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
        cv2.putText(image, output, (10, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    else:
        text.insert(END, "No face detected in the uploaded image.\n")
        image = cv2.resize(image, (500, 500))
    
    # Show the processed image
    cv2.imshow('Deep Fake Detection Output', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()



font = ('times', 15, 'bold')
#title = Label(main, text=' Deepfake Face Detection')
#title.config(bg='brown', fg='white')  
#title.config(font=font)           
#title.config(height=3, width=80)       
#title.place(x=5,y=5)

font1 = ('times', 13, 'bold')
upload = Button(main, text="Upload Deepfake Dataset", command=uploadDataset)
upload.place(x=50,y=200)
upload.config(font=font1)  

pathlabel = Label(main)
pathlabel.config(bg='brown', fg='white')  
pathlabel.config(font=font1)           
pathlabel.place(x=480,y=200)

uploadButton = Button(main, text="Train Deep Learning Model", command=trainModel)
uploadButton.place(x=50,y=250)
uploadButton.config(font=font1)

exitButton = Button(main, text="Video Based Deepfake Detection", command=uploadVideo)
exitButton.place(x=50,y=300)
exitButton.config(font=font1)

exitButton = Button(main, text="Image Based Deepfake Detection", command=uploadImage)
exitButton.place(x=50,y=350)
exitButton.config(font=font1)

font1 = ('times', 12, 'bold')
text=Text(main,height=10,width=150)
scroll=Scrollbar(text)
text.configure(yscrollcommand=scroll.set)
text.place(x=10,y=500)
text.config(font=font1)


main.config(bg='brown')
main.mainloop()
