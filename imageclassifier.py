import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from PIL import Image
import scipy
# Set the path to your dataset
train_data_dir = '/Users/wesle/Downloads/Ocean-Shoreline-recognition/images/train'
validation_data_dir = '/Users/wesle/Downloads/Ocean-Shoreline-recognition/images/validation'

# Parameters
img_width, img_height = 224, 224
batch_size = 32
epochs = 10

# Data Augmentation
train_datagen = ImageDataGenerator(
    rescale=1./255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True
)

validation_datagen = ImageDataGenerator(rescale=1./255)

# Load and prepare the training data
train_generator = train_datagen.flow_from_directory(
    train_data_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='binary'
)

# Load and prepare the validation data
validation_generator = validation_datagen.flow_from_directory(
    validation_data_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='binary'
)
print("Number of samples in the training dataset:", train_generator.samples)
print("Batch size:", batch_size)

# Check if the dataset contains any samples
if train_generator.samples == 0:
    print("Error: The training dataset contains no samples.")
else:
    # Calculate the number of steps_per_epoch based on the training dataset size and batch size
    steps_per_epoch = train_generator.samples // batch_size
    # Build a simple CNN model
    model = models.Sequential()
    model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(img_width, img_height, 3)))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(128, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Flatten())
    model.add(layers.Dense(256, activation='relu'))
    model.add(layers.Dropout(0.5))
    model.add(layers.Dense(1, activation='sigmoid'))

    # Compile the model
    model.compile(
        optimizer='adam',
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    print(train_generator.samples)
    # Train the model
    model.fit(
        train_generator,
        
        epochs=epochs,
        validation_data=validation_generator,
        validation_steps=1
    )

    # Save the trained model
    model.save('ocean_land_classifier_model.h5')

    # Now, you can use the trained model to make predictions on new images
