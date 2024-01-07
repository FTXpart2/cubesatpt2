import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model
import numpy as np
import matplotlib.pyplot as plt

# Load the trained model
model = load_model('ocean_land_classifier_model.h5')

# Define the path to the test image
test_image_path = 'path/to/image.jpg' #Insert path to image

# Load and preprocess the test image
img = image.load_img(test_image_path, target_size=(224, 224))
img_array = image.img_to_array(img)
img_array = np.expand_dims(img_array, axis=0)
img_array /= 255.0

# Make predictions
predictions = model.predict(img_array)

# Print the predicted class
if predictions[0][0] > 0.5:
    print("Prediction: Land")
else:
    print("Prediction: Ocean")

# Display the test image
plt.imshow(img)
plt.axis('off')
plt.show()
