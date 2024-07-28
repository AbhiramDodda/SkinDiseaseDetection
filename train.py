import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
from keras.models import Sequential, Model
from keras.optimizers import Adam
from keras.applications import EfficientNetB7
from keras.callbacks import ReduceLROnPlateau
from keras.utils import image_dataset_from_directory
from keras.layers import Conv2D, MaxPool2D, Flatten, Dense, Dropout, BatchNormalization
from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing import image
from sklearn.metrics import confusion_matrix,accuracy_score, classification_report
import keras
import random

''' Block to visualize class frequencies before oversampling and undersampling '''
data_path = 'data\ISIC_Labelled'
classes = os.listdir(data_path) # names of folders are the names of classes
class_counts = [len(os.listdir(data_path + '/' + x)) for x in classes] # length of respective classes is the count of images available for the respective class
#print(class_counts)
plt.figure(figsize=(20, 6))
bars = plt.bar(classes, class_counts, color = ['#FF5733', '#33FF57', '#3357FF', '#FF33A6', '#FF8633', '#33FFF3', '#FF3333', '#8633FF']) # 8 random colors chosen for better visualisation
plt.xlabel('Classes')
plt.ylabel('Counts')
plt.title('Class Counts of 8 Classes')
# Code to display count of each class on top of respective bar
for bar in bars:
    height = bar.get_height()
    plt.text(
        bar.get_x() + bar.get_width() / 2, 
        height, 
        f'{height}', 
        ha='center', 
        va='bottom'
    )

def oversampler(class_name, class_count):
    datagen = ImageDataGenerator(
        rotation_range=40,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest'
    )

    curr_dir = data_path + '/' + class_name
    save_dir = curr_dir
    images = os.listdir(curr_dir)
    img = plt.imread(curr_dir + '/' + random.choice(images))
    img = np.expand_dims(img, 0)
    

    cnt = class_count
    target = 4000
    for batch in datagen.flow(img, batch_size=1, save_to_dir=save_dir, save_prefix='aug', save_format='jpg'):
        cnt += 1
        if cnt >= target:
            break
        img = plt.imread(curr_dir + '/' + random.choice(images))
        img = np.expand_dims(img, 0)
        

def undersampler(class_name, class_count):
    target_count = 4000
    curr_dir = data_path + '/' + class_name
    images = os.listdir(curr_dir)
    images_to_delete = random.sample(images, class_count - target_count)
    for image in images_to_delete:
        img_path = os.path.join(curr_dir, image)
        os.remove(img_path)

for i in range(len(classes)):
    if class_counts[i] < 4000:
        oversampler(classes[i], class_counts[i])
    else:
        undersampler(classes[i], class_counts[i])

''' Block to visualize class frequencies after oversampling and undersampling '''
data_path = 'data\ISIC_Labelled'
classes = os.listdir(data_path) # names of folders are the names of classes
class_counts = [len(os.listdir(data_path + '/' + x)) for x in classes] # length of respective classes is the count of images available for the respective class
#print(class_counts)
plt.figure(figsize=(20, 6))
bars = plt.bar(classes, class_counts, color = ['#FF5733', '#33FF57', '#3357FF', '#FF33A6', '#FF8633', '#33FFF3', '#FF3333', '#8633FF']) # 8 random colors chosen for better visualisation
plt.xlabel('Classes')
plt.ylabel('Counts')
plt.title('Class Counts of 8 Classes')
# Code to display count of each class on top of respective bar
for bar in bars:
    height = bar.get_height()
    plt.text(
        bar.get_x() + bar.get_width() / 2, 
        height, 
        f'{height}', 
        ha='center', 
        va='bottom'
    )

import random
import shutil

train_path = "train"
val_path = "val"
test_path = "test"

os.makedirs(train_path, exist_ok=True)
os.makedirs(val_path, exist_ok=True)
os.makedirs(test_path, exist_ok=True)

train_ratio = 0.7  # 70% for training
val_ratio = 0.15   # 15% for validation
test_ratio = 0.15  # 15% for testing

for class_folder in os.listdir(data_path):
    class_path = os.path.join(data_path, class_folder)
    
    train_class_path = os.path.join(train_path, class_folder)
    val_class_path = os.path.join(val_path, class_folder)
    test_class_path = os.path.join(test_path, class_folder)
    
    os.makedirs(train_class_path, exist_ok=True)
    os.makedirs(val_class_path, exist_ok=True)
    os.makedirs(test_class_path, exist_ok=True)   
    images = os.listdir(class_path)
    random.shuffle(images)
    num_images = len(images)
    num_train = int(num_images * train_ratio)
    num_val = int(num_images * val_ratio)
    num_test = num_images - num_train - num_val
    
    train_images = images[:num_train]
    val_images = images[num_train:num_train+num_val]
    test_images = images[num_train+num_val:]
    
    for image in train_images:
        shutil.copy(os.path.join(class_path, image), os.path.join(train_class_path, image))
    
    for image in val_images:
        shutil.copy(os.path.join(class_path, image), os.path.join(val_class_path, image))
    
    for image in test_images:
        shutil.copy(os.path.join(class_path, image), os.path.join(test_class_path, image))

print("Dataset split completed successfully.")

val_path = "val"
train_path = "train"
img_size =(256,256)
batch_size = 32
train_dataset = image_dataset_from_directory(
    train_path,
    image_size=img_size,
    batch_size=batch_size,
    seed=123
)
class_names = train_dataset.class_names
val_dataset = image_dataset_from_directory(
    val_path,
    image_size=img_size,
    batch_size=batch_size,
    seed=42
)

for image_batch, labels_batch in train_dataset.take(1):
    print(image_batch.shape)
    print(labels_batch.numpy())
plt.figure(figsize=(20, 20))
from PIL import Image
i = 0
for image_class in os.listdir(train_path):
    class_path = os.path.join(train_path, image_class)
    image_file = os.listdir(class_path)[0]
    image_path = os.path.join(class_path, image_file)
    image = Image.open(image_path)
    ax = plt.subplot(3, 4, i + 1)
    plt.imshow(image)
    plt.title(image_class)
    plt.axis("off")
    i += 1

import numpy as np
import cv2

def stain_normalization(image, target_mean=np.array([196, 154, 122]), target_std=np.array([33, 11, 13])):
    # Convert image to LAB color space
    image_lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
    
    # Compute mean and standard deviation of each channel
    image_mean = np.mean(image_lab, axis=(0, 1))
    image_std = np.std(image_lab, axis=(0, 1))
    
    # Perform stain normalization
    image_lab[:, :, 0] = (image_lab[:, :, 0] - image_mean[0]) * (target_std[0] / image_std[0]) + target_mean[0]
    image_lab[:, :, 1] = (image_lab[:, :, 1] - image_mean[1]) * (target_std[1] / image_std[1]) + target_mean[1]
    image_lab[:, :, 2] = (image_lab[:, :, 2] - image_mean[2]) * (target_std[2] / image_std[2]) + target_mean[2]
    
    # Convert back to RGB color space
    normalized_image = cv2.cvtColor(image_lab, cv2.COLOR_LAB2RGB)
    
    return normalized_image

# Example usage
input_image = cv2.imread('D:\\train\\Actinic keratosis\\ISIC_0070200.jpg')
#print(input_image)
input_image = cv2.cvtColor(input_image, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB
normalized_image = stain_normalization(input_image)

# Display original and normalized images
import matplotlib.pyplot as plt

plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.imshow(input_image)
plt.title('Original Image')
plt.axis('off')

plt.subplot(1, 2, 2)
plt.imshow(normalized_image)
plt.title('Normalized Image')
plt.axis('off')

plt.show()

from keras.applications.efficientnet import preprocess_input

def preprocess_image(image_path, img_size=(256, 256)):
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  
    normalized_image = stain_normalization(image)

    resized_image = cv2.resize(normalized_image, img_size)
 
    preprocessed_image = preprocess_input(resized_image)
    
    return preprocessed_image

val_path = "val"
train_path = "train"



# Feature extractor
base_model = keras.applications.EfficientNetB3(
    input_shape=(256, 256, 3), 
    include_top=False,             
    weights='imagenet',
    pooling='max'
) 

for layer in base_model.layers:
    layer.trainable = False

# Build model
inputs = base_model.input
x = BatchNormalization()(base_model.output)
x = Dense(1024, activation='relu')(x)
x = Dense(512, activation='relu')(x)
x = Dense(512, activation='relu')(x)
x = Dense(256, activation='relu')(x)
x = Flatten()(x)
outputs = Dense(8, activation='softmax')(x)
model = Model(inputs=inputs, outputs=outputs)

# Compile model
model.compile(optimizer=Adam(learning_rate=0.001), loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Train model
ep = 50
history = model.fit(train_dataset,
                    validation_data=val_dataset,
                    epochs=ep)

# Save model
model.save('SkinDiseaseWeights.h5')

from sklearn.metrics import precision_score, recall_score, confusion_matrix, classification_report, accuracy_score, f1_score
import tensorflow as tf

test_path = 'test'

test_dataset = image_dataset_from_directory(
    test_path,
    image_size=img_size,
    batch_size=batch_size,
    seed=42
)

labels = []
predictions = []
for x,y in test_dataset:
    labels.append(list(y.numpy().astype("uint8")))
    predictions.append(tf.argmax(model.predict(preprocess_input(x)),1).numpy().astype("uint8"))
import itertools
predictions = list(itertools.chain.from_iterable(predictions))
labels = list(itertools.chain.from_iterable(labels))
print("Train Accuracy  : {:.2f} %".format(history.history['accuracy'][-1]*100))
print("Test Accuracy   : {:.2f} %".format(accuracy_score(labels, predictions) * 100))
print("Precision Score : {:.2f} %".format(precision_score(labels, predictions, average='micro') * 100))
print("Recall Score    : {:.2f} %".format(recall_score(labels, predictions, average='micro') * 100))
print('F1 score:', f1_score(labels, predictions, average='micro'))

plt.figure(figsize = (20,5))
plt.subplot(1,2,1)
plt.title("Train and Validation Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.plot(history.history['loss'],label="Train Loss")
plt.plot(history.history['val_loss'], label="Validation Loss")
plt.xlim(0, 10)
plt.ylim(0.0,1.0)
plt.legend()

plt.subplot(1,2,2)
plt.title("Train and Validation Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.plot(history.history['accuracy'], label="Train Accuracy")
plt.plot(history.history['val_accuracy'], label="Validation Accuracy")
plt.xlim(0, 9.25)
plt.ylim(0.0,1.0)
plt.legend()
plt.tight_layout()

import sklearn
plt.figure(figsize= (20,5))
cm = confusion_matrix(labels, predictions)
disp = sklearn.metrics.ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=list(range(1,9)))
fig, ax = plt.subplots(figsize=(15,15))
disp.plot(ax=ax,colorbar= False,cmap = 'YlGnBu')
plt.title("Confusion Matrix")
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.show()