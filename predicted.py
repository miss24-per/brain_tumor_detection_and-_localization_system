import tensorflow as tf
import numpy as np
import cv2
import matplotlib.pyplot as plt

# Load model
model = tf.keras.models.load_model("brain_tumor_model.keras")

# Classes
classes = ['glioma', 'meningioma', 'notumor', 'pituitary']

img_path = r"C:\Users\sachi\OneDrive\Desktop\brain_tumor_project\dataset1\DATASET\classification\Testing\glioma\enh_Te-glTr_0000.jpg"

# Read image
img = cv2.imread(img_path)

img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# Resize
img_resized = cv2.resize(img, (224,224))

# Normalize
img_resized = img_resized / 255.0

# Expand dimensions
input_img = np.expand_dims(img_resized, axis=0)

# Predict
prediction = model.predict(input_img)

predicted_class = classes[np.argmax(prediction)]

confidence = np.max(prediction)

# Display
plt.imshow(img)
plt.title(f"{predicted_class} ({confidence:.2f})")
plt.axis("off")
plt.show()