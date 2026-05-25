import tensorflow as tf
import numpy as np
import cv2
import matplotlib.pyplot as plt

# Load trained model
model = tf.keras.models.load_model("brain_tumor_model.keras")

# Image path
img_path = r"C:\Users\sachi\OneDrive\Desktop\brain_tumor_project\dataset1\DATASET\classification\Testing\glioma\enh_Te-glTr_0000.jpg"

# Read image
img = cv2.imread(img_path)

# Convert BGR to RGB
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# Resize image
img_resized = cv2.resize(img, (224,224))

# Normalize
img_resized = img_resized / 255.0

# Expand dimensions
input_image = np.expand_dims(img_resized, axis=0)

# Last convolutional layer
last_conv_layer = model.get_layer("Conv_1")

# Create Grad-CAM model
grad_model = tf.keras.models.Model(
    inputs=model.inputs,
    outputs=[last_conv_layer.output, model.output]
)

# Predict class
predictions = model.predict(input_image)

predicted_class = np.argmax(predictions[0])

print("Predicted Class Index:", predicted_class)

# Gradient calculation
with tf.GradientTape() as tape:

    conv_outputs, predictions = grad_model(input_image)

    loss = predictions[:, predicted_class]

# Get gradients
grads = tape.gradient(loss, conv_outputs)

# Global average pooling
pooled_grads = tf.reduce_mean(grads, axis=(0,1,2))

# Remove batch dimension
conv_outputs = conv_outputs[0]

# Weight feature maps
heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]

heatmap = tf.squeeze(heatmap)

# ReLU activation
heatmap = np.maximum(heatmap, 0)

# Normalize heatmap
heatmap = heatmap / np.max(heatmap)

# Resize heatmap
heatmap = cv2.resize(
    heatmap,
    (img.shape[1], img.shape[0])
)

# Convert to uint8
heatmap = np.uint8(255 * heatmap)

# Apply color map
heatmap = cv2.applyColorMap(
    heatmap,
    cv2.COLORMAP_JET
)

# Overlay heatmap on image
superimposed_img = cv2.addWeighted(
    cv2.cvtColor(img, cv2.COLOR_RGB2BGR),
    0.6,
    heatmap,
    0.4,
    0
)

# Convert back to RGB
superimposed_img = cv2.cvtColor(
    superimposed_img,
    cv2.COLOR_BGR2RGB
)

# Display results
plt.figure(figsize=(12,5))

# Original MRI
plt.subplot(1,2,1)

plt.imshow(img)

plt.title("Original MRI")

plt.axis("off")

# Grad-CAM output
plt.subplot(1,2,2)

plt.imshow(superimposed_img)

plt.title("Tumor Localization using Grad-CAM")

plt.axis("off")

plt.show()