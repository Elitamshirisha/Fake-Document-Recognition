import numpy as np
import matplotlib.pyplot as plt
import os
import cv2
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Model
from tensorflow.keras import Input
from tensorflow.keras.layers import Dense, Dropout, Conv2D, BatchNormalization, GlobalAveragePooling2D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.applications import DenseNet121
from tensorflow.keras.utils import to_categorical
from numpy import argmax
from sklearn.metrics import confusion_matrix, classification_report
from mlxtend.plotting import plot_confusion_matrix
import easygui
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping

print("**********************************************")
print("Project Title  --Fake Document Recognition   ")
print("**********************************************")
print()

#==============================Input Data======================================
path = 'Dataset/'

# categories
categories = ['Forgery','Original']
valid_extensions = ['.png', '.jpg', '.jpeg']

# let's display some of the pictures
for category in categories:
    fig, axes = plt.subplots(2, 2)
    fig.suptitle(category)
    fig.patch.set_facecolor('xkcd:white')
    for k, v in enumerate(os.listdir(path+category)[:4]):
        img_path = os.path.join(path, category, v)
        img = cv2.imread(img_path)
        if img is not None:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) # Convert to RGB for matplotlib
            axes[k//2, k%2].axis('off')
            axes[k//2, k%2].imshow(img)
        else:
            print(f"Error loading image: {img_path}")
    plt.show()

shape0 = []
shape1 = []

for category in categories:
    for files in os.listdir(path+category):
        img_path = os.path.join(path, category, files)
        img = cv2.imread(img_path)
        if img is not None:
            shape0.append(img.shape[0])
            shape1.append(img.shape[1])
        else:
            print(f"Error loading image: {img_path}")
    if shape0 and shape1:
        print(category, ' => height min : ', min(shape0), 'width min : ', min(shape1))
        print(category, ' => height max : ', max(shape0), 'width max : ', max(shape1))
    shape0 = []
    shape1 = []

# initialize the data and labels
data = []
labels = []
imagePaths = []
HEIGHT = 128  # Increased image size
WIDTH = 128   # Increased image size
N_CHANNELS = 3

# grab the image paths and randomly shuffle them
for k, category in enumerate(categories):
    for f in os.listdir(path+category):
        imagePaths.append([os.path.join(path, category, f), k])

import random
random.shuffle(imagePaths)
print("First 10 image paths:", imagePaths[:10])

# loop over the input images
for imagePath in imagePaths:
    try:
        # load the image, resize the image to be HEIGHT * WIDTH pixels (ignoring aspect ratio) and store the image in the data list
        image = cv2.imread(imagePath[0])
        if image is not None:
            image = cv2.resize(image, (WIDTH, HEIGHT))
            data.append(image)

            # extract the class label from the image path and update the
            # labels list
            label = imagePath[1]
            labels.append(label)
        else:
            print(f"Error loading image: {imagePath[0]}")
    except Exception as e:
        print(f"Error processing image: {imagePath[0]} - {e}")

# scale the raw pixel intensities to the range [0, 1]
data = np.array(data, dtype="float") / 255.0
labels = np.array(labels)

# Let's check everything is ok
fig, axes = plt.subplots(2, 2)
fig.suptitle("Sample Input")
fig.patch.set_facecolor('xkcd:white')
for i in range(min(4, len(data))):
    axes[i//2, i%2].imshow(cv2.cvtColor(data[i].astype(np.float32), cv2.COLOR_BGR2RGB))
    axes[i//2, i%2].axis('off')
plt.show()

# partition the data into training and testing splits using 80% of
# the data for training and the remaining 20% for testing
(trainX, testX, trainY, testY) = train_test_split(data, labels, test_size=0.2, random_state=42)
# Preprocess class labels
trainY = to_categorical(trainY, 2)
testY_categorical = to_categorical(testY, 2) # Convert test labels to categorical

print("TrainX shape:", trainX.shape)
print("TestX shape:", testX.shape)
print("TrainY shape:", trainY.shape)
print("TestY shape:", testY_categorical.shape)

#================================Classification================================
'''DENSENET121 with Data Augmentation and Fine-tuning'''

def build_fine_tuned_densenet(height, width, channels, num_classes):
    densenet = DenseNet121(weights='imagenet', include_top=False, input_shape=(height, width, channels))

    # Unfreeze the last few convolutional blocks (adjust the number based on experimentation)
    for layer in densenet.layers[:300]:
        layer.trainable = False
    for layer in densenet.layers[300:]:
        layer.trainable = True

    input_tensor = Input(shape=(height, width, channels))
    x = densenet(input_tensor)
    x = GlobalAveragePooling2D()(x)
    x = BatchNormalization()(x)
    x = Dropout(0.5)(x)
    x = Dense(256, activation='relu')(x)
    x = BatchNormalization()(x)
    x = Dropout(0.5)(x)
    output = Dense(num_classes, activation='softmax', name='root')(x)
    model = Model(input_tensor, output)
    model.compile(optimizer=Adam(learning_rate=1e-4),  # Lower learning rate for fine-tuning
                  loss='binary_crossentropy',
                  metrics=['accuracy'])
    model.summary()
    return model

model = build_fine_tuned_densenet(HEIGHT, WIDTH, N_CHANNELS, len(categories))

# Data Augmentation
train_datagen = ImageDataGenerator(
    rotation_range=20,
    width_shift_range=0.1,
    height_shift_range=0.1,
    shear_range=0.1,
    zoom_range=0.1,
    horizontal_flip=True,
    fill_mode='nearest'
)

test_datagen = ImageDataGenerator()

train_generator = train_datagen.flow(trainX, trainY, batch_size=32)
validation_generator = test_datagen.flow(testX, testY_categorical, batch_size=32)

# Early stopping to prevent overfitting
early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

history = model.fit(
    train_generator,
    steps_per_epoch=len(trainX) // 32,
    epochs=50,  # Increased number of epochs
    verbose=1,
    validation_data=validation_generator,
    validation_steps=len(testX) // 32,
    callbacks=[early_stopping]
)

tf.keras.models.save_model(model,'Model_fine_tuned.h5')

#Plotting the accuracy and loss
epochs = range(len(history.history['accuracy']))

plt.figure(figsize=(12, 5))

# Plot training & validation accuracy values
plt.subplot(1, 2, 1)
plt.plot(epochs, history.history['accuracy'], label='Train Accuracy')
plt.plot(epochs, history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Model Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()

# Plot training & validation loss values
plt.subplot(1, 2, 2)
plt.plot(epochs, history.history['loss'], label='Train Loss')
plt.plot(epochs, history.history['val_loss'], label='Validation Loss')
plt.title('Model Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

plt.tight_layout()
plt.show()

#=============================Analytic Results=================================
pred = model.predict(testX)
predictions = argmax(pred, axis=1)
print('Classification Report')
cr = classification_report(testY, predictions, target_names=categories)
print(cr)
print('Confusion Matrix')
cm = confusion_matrix(testY, predictions)
print(cm)
#Confusion Matrix Plot
plt.figure()
plot_confusion_matrix(cm, figsize=(8, 8), class_names = categories, show_normed = True, colorbar=True)
plt.title("Model Confusion Matrix")
plt.style.use("ggplot")
plt.show()
#=================================Prediction===================================
def predict_image(model, image_path, target_size=(128, 128)):
    try:
        test_image_o = cv2.imread(image_path)
        if test_image_o is not None:
            test_image = cv2.resize(test_image_o, target_size)
            test_image = cv2.cvtColor(test_image, cv2.COLOR_BGR2RGB) # Ensure RGB for prediction display
            test_data = np.array(test_image, dtype="float") / 255.0
            test_data = np.expand_dims(test_data, axis=0) # Add batch dimension
            pred = model.predict(test_data)
            predictions = argmax(pred, axis=1) # return to label
            confidence = pred[0][predictions[0]] * 100
            return categories[predictions[0]], confidence, test_image_o
        else:
            return None, None, None
    except Exception as e:
        print(f"Error predicting image: {e}")
        return None, None, None

if __name__ == '__main__':
    loaded_model = tf.keras.models.load_model('Model_fine_tuned.h5')
    Image = easygui.fileopenbox(title="Select an image for prediction")
    if Image:
        prediction, confidence, original_image = predict_image(loaded_model, Image)
        if prediction:
            print(f'Prediction: {prediction} (Confidence: {confidence:.2f}%)')
            plt.figure()
            plt.imshow(cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB))
            plt.title(f'Predicted: {prediction} (Confidence: {confidence:.2f}%)')
            plt.axis('off')
            plt.show()
        else:
            print("Could not process the selected image.")