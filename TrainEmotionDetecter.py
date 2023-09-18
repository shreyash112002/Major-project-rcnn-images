# Import required packages
import cv2
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Dense, Dropout, Flatten
from keras.optimizers import Adam
from keras.preprocessing.image import ImageDataGenerator

# Initialize image data generator with rescaling and validation split
data_gen = ImageDataGenerator(rescale=1./255, validation_split=0.2)

# Load and preprocess all images with the updated target_size
image_data = data_gen.flow_from_directory(
        'Student-Engagement-Dataset',
        target_size=(224, 224),  # Reduce image size to 224x224
        batch_size=64,
        color_mode="rgb",
        class_mode='categorical',
        subset='training')  # Use 'training' subset for training data

# Create a separate generator for validation (testing) data
validation_data = data_gen.flow_from_directory(
        'Student-Engagement-Dataset',
        target_size=(224, 224),  # Reduce image size to 224x224
        batch_size=64,
        color_mode="rgb",
        class_mode='categorical',
        subset='validation')  # Use 'validation' subset for testing data

# Create model structure
emotion_model = Sequential()
emotion_model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(224, 224, 3)))
emotion_model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
emotion_model.add(MaxPooling2D(pool_size=(2, 2)))
emotion_model.add(Dropout(0.25))
emotion_model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
emotion_model.add(MaxPooling2D(pool_size=(2, 2)))
emotion_model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
emotion_model.add(MaxPooling2D(pool_size=(2, 2)))
emotion_model.add(Dropout(0.25))
emotion_model.add(Flatten())
emotion_model.add(Dense(1024, activation='relu'))
emotion_model.add(Dropout(0.5))
# Update the number of units in the output layer to match the number of classes (6)
emotion_model.add(Dense(6, activation='softmax'))

cv2.ocl.setUseOpenCL(False)

# Compile the model with the 'learning_rate' parameter
emotion_model.compile(loss='categorical_crossentropy', optimizer=Adam(learning_rate=0.0001), metrics=['accuracy'])

# Train the neural network/model using the training data
emotion_model_info = emotion_model.fit(
        image_data,
        steps_per_epoch=len(image_data),
        epochs=20,
        validation_data=validation_data,
        validation_steps=len(validation_data))

# Save model structure in JSON file
model_json = emotion_model.to_json()
with open("emotion_model.json", "w") as json_file:
    json_file.write(model_json)

# Save trained model weights in .h5 file
emotion_model.save_weights('emotion_model.h5')
