

import os
import sys
import numpy as np
import operator
import pickle
from keras.models import Sequential, load_model

from keras.preprocessing import image as image_utils
import numpy
from keras.preprocessing import image

from keras.preprocessing.image import img_to_array

import warnings
warnings.filterwarnings("ignore")

def prediction_image(test_image):
    try:

        data = []
       
       
        model_path = 'vgg19_model.h5'
        model = load_model(model_path)

        test_image = image.load_img(test_image, target_size=(128, 128))
        test_image = img_to_array(test_image)
        test_image = np.expand_dims(test_image, axis=0)
        test_image /= 255
        prediction = model.predict(test_image)
        lb = pickle.load(open('label_transform_vgg19.pkl', 'rb'))
        prediction_result=lb.inverse_transform(prediction)[0]

        print(prediction_result)

        return prediction_result





    except Exception as e:
        print("Error=", e)
        tb = sys.exc_info()[2]
        print("LINE NO: ", tb.tb_lineno)

# testimage="006.jpg"
testimage="C://Users//akshi//Music//Detection//dataset//Glaucoma Negative//007.jpg"
prediction_image(testimage)

