import os
import numpy as np
import pickle

from keras.applications.vgg19 import VGG19
import keras
from os import listdir
from sklearn.preprocessing import LabelBinarizer
from keras.models import Sequential
#from keras.layers.normalization import BatchNormalization
from tensorflow.keras.layers import BatchNormalization
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.layers.core import Activation, Flatten, Dropout, Dense
from keras import backend as K
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import Adam
from keras.preprocessing import image
from keras.preprocessing.image import img_to_array
# from tensorflow.keras.utils import img_to_array,load_img
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.model_selection import train_test_split
import matplotlib.pyplot  as plt
from keras.preprocessing import image

from sklearn.metrics import f1_score, precision_score, accuracy_score, recall_score
import sys

def build_vgg19():
    try:

        EPOCHS = 10

        BS = 16
        
        image_size = 0
        width = 128
        height = 128
        depth = 3
        print("[INFO] Loading Training dataset images...")
        DIRECTORY = "..\\Detection\\dataset"
        CATEGORIES = ['Glaucoma Negative', 'Glaucoma Positive']

        data = []
        clas = []

        for category in CATEGORIES:
            print(category)
            path = os.path.join(DIRECTORY, category)
            print(path)
            for img in os.listdir(path):
                img_path = os.path.join(path, img)
                img = image.load_img(img_path, target_size=(128, 128))
                img = img_to_array(img)
                # img = img / 255
                data.append(img)
                clas.append(category)

        label_binarizer = LabelBinarizer()
        image_labels = label_binarizer.fit_transform(clas)
        pickle.dump(label_binarizer, open('label_transform_vgg19.pkl', 'wb'))
        n_classes = len(label_binarizer.classes_)
        print(n_classes)
        np_image_list = np.array(data, dtype=np.float16) / 225.0

        x_train, x_test, y_train, y_test = train_test_split(np_image_list, image_labels, test_size=0.2, random_state=27)
        

        base_model = VGG19(include_top=False, input_shape=(128, 128, 3))
        base_model.trainable = False
        classifier = keras.models.Sequential()
        classifier.add(base_model)
        classifier.add(Flatten())
        classifier.add(Dense(2, activation='softmax'))

        #opt=Adam(lr=INIT_LR,decay=INIT_LR/EPOCHS)

        classifier.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

        print("[INFO] training network...")

        aug = ImageDataGenerator(
            rotation_range=30, width_shift_range=0.1,
            height_shift_range=0.1, shear_range=0.2,
            zoom_range=0.2, horizontal_flip=True,
            fill_mode="nearest")

        history = classifier.fit_generator(
            aug.flow(x_train, y_train, batch_size=BS),
            validation_data=(x_test, y_test),
            steps_per_epoch=len(x_train) // BS,
            epochs=EPOCHS, verbose=1
        )

        pred=classifier.predict(x_test)
        pred_y=pred.argmax(axis=1)

        accuracy=accuracy_score(y_test,pred_y)*100

        pre=precision_score(y_test,pred_y)*100

        f1score=f1_score(y_test,pred_y)*100

        recall=recall_score(y_test,pred_y)*100
        
        print("VGG19")
        print("Accuracy=",accuracy)

        print("Precision=",pre)

        print("Recall=",recall)

        print("F1-Score=",f1score)





        acc = history.history['acc']
        val_acc = history.history['val_acc']
        loss = history.history['loss']
        val_loss = history.history['val_loss']
        epochs = range(1, len(acc) + 1)
        # Train and validation accuracy
        plt.plot(epochs, acc, 'b', label='Training accurarcy')
        plt.plot(epochs, val_acc, 'r', label='Validation accurarcy')
        plt.title('Training and Validation accurarcy')
        plt.legend()
        plt.savefig('vgg19_accuracy.png')

        plt.figure()
        # Train and validation loss
        plt.plot(epochs, loss, 'b', label='Training loss')
        plt.plot(epochs, val_loss, 'r', label='Validation loss')
        plt.title('Training and Validation loss')
        plt.legend()
        plt.savefig('vgg19_loss.png')
        plt.show()




        print("Training Completed..!")

        # save the model to disk
        print("[INFO] Saving model...")
        #classifier.save('vgg19_model.h5')

    except Exception as e:
        print("Error=", e)
        tb = sys.exc_info()[2]
        print(tb)
        print(tb.tb_lineno)

build_vgg19()
