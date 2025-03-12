# train.py

import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from create_cnn_model import create_cnn_model

# Load dataset
dataset_dir = 'path/to/dataset'
image_size = (224, 224)

# Load training data
train_datagen = ImageDataGenerator(rescale=1./255)
train_generator = train_datagen.flow_from_directory(
    dataset_dir + '/train',
    target_size=image_size,
    batch_size=32,
    class_mode='categorical'
)

# Load validation data
validation_datagen = ImageDataGenerator(rescale=1./255)
validation_generator = validation_datagen.flow_from_directory(
    dataset_dir + '/validation',
    target_size=image_size,
    batch_size=32,
    class_mode='categorical'
)

# Create model
input_shape = (image_size[0], image_size[1], 3)
model = create_cnn_model(input_shape)

# Train model
history = model.fit(
    train_generator,
    epochs=10,
    validation_data=validation_generator
)

# Save model
model.save('gesture_recognition_model.h5')