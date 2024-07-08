from keras.models import load_model
from sklearn.metrics import precision_score, recall_score, f1_score
from keras.preprocessing import image
import tensorflow as tf
import numpy
import keras
from PIL import Image
import cv2



model = load_model('SkinDiseaseWeights.h5')
def diseaesePred(imagePath):
    images = imagePath
    img = image.load_img(images)
    #img = cv2.resize(img, (256,256))
    img_array = keras.preprocessing.image.img_to_array(img)
    img_array = cv2.resize(img_array, (256, 256))
    img_array = tf.expand_dims(img_array, 0)
    predictions = model.predict(img_array)
    class_names = ['Actinic keratosis', 'Basal cell carcinoma', 'Benign keratosis', 'Dermatofibroma', 
                   'Melanocytic nevus', 'Melanoma', 'Squamous cell carcinoma', 'Vascular lesion']
    predicted_class = class_names[numpy.argmax(predictions[0])]
    confidence = round(100*(numpy.max(predictions[0])), 2)
    print(predicted_class, confidence)
    return predicted_class, confidence