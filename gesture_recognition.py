import tensorflow as tf
from tensorflow.keras import layers, models

# Define the CNN model for gesture classification
def create_cnn_model(input_shape):
    model = models.Sequential()  # Initialize a Sequential model

    # First Convolutional Layer
    model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape))  # Conv2D layer with 32 filters and ReLU activation
    model.add(layers.MaxPooling2D((2, 2)))  # Max pooling layer to downsample the feature map

    # Second Convolutional Layer
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))  # Conv2D layer with 64 filters
    model.add(layers.MaxPooling2D((2, 2)))  # Max pooling again

    # Third Convolutional Layer  
    model.add(layers.Conv2D(128, (3, 3), activation='relu'))  # Conv2D layer with 128 filters
    model.add(layers.MaxPooling2D((2, 2)))  # Max pooling again

    # Flatten the feature map to a 1D array for fully connected layers
    model.add(layers.Flatten())

    # Fully connected layer
    model.add(layers.Dense(128, activation='relu'))  # Fully connected layer with 128 neurons
    model.add(layers.Dense(64, activation='relu'))  # Another fully connected layer with 64 neurons

    # Output layer with 26 neurons (assuming we are recognizing A-Z gestures)
    model.add(layers.Dense(26, activation='softmax'))  # Softmax output layer for multi-class classification

    # Compile the model with Adam optimizer and categorical cross-entropy loss function
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

# Example input shape for hand gesture images (can be adjusted)
input_shape = (64, 64, 3)  # Image input size: 64x64 with 3 color channels (RGB)
model = create_cnn_model(input_shape)  # Create the CNN model
model.summary()  # Print the model summary to check its architecture
# Model training will require a dataset of hand gestures (not covered in detail here)
