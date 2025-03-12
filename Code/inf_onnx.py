import onnxruntime as ort
import numpy as np
import cv2
import matplotlib.pyplot as plt

# Load ONNX model
session = ort.InferenceSession("fundus.onnx", providers=["CPUExecutionProvider"])

# Padding function to ensure divisibility by 32
def pad_to_divisibility(image, divisor=32):
    """Pad image to make its height and width divisible by the given divisor."""
    h, w, c = image.shape
    pad_h = (divisor - h % divisor) % divisor
    pad_w = (divisor - w % divisor) % divisor

    # Pad evenly on all sides
    top = pad_h // 2
    bottom = pad_h - top
    left = pad_w // 2
    right = pad_w - left

    padded_image = cv2.copyMakeBorder(image, top, bottom, left, right, cv2.BORDER_REFLECT_101)
    return padded_image, (top, bottom, left, right)

# Load image
image_path = r"C:\Users\iamgo\Downloads\fundus_inf\i vision eye hospital\GLAUCOMA\2017 973_20231225135005557.jpg"  # Update this path
image = cv2.imread(image_path, cv2.IMREAD_COLOR)

if image is None:
    print("Error: Image not found!")
    exit()

print(f"Original image size: {image.shape}")

# Pad the image
padded_image, padding = pad_to_divisibility(image)
print(f"Padded image size: {padded_image.shape}")

# Preprocess image for ONNX model
x = padded_image.astype(np.float32) / 255.0  # Normalize
x = np.transpose(x, (2, 0, 1))  # Change to (C, H, W)
x = np.expand_dims(x, axis=0)  # Add batch dimension

# Run inference
input_name = session.get_inputs()[0].name
output_name = session.get_outputs()[0].name
output = session.run([output_name], {input_name: x})[0]

# Post-process output
output = 1 / (1 + np.exp(-output))  # Sigmoid activation
output = (output > 0.3).astype(np.uint8) * 255  # Threshold segmentation map

# Remove padding
top, bottom, left, right = padding
output = output[0, 0, top:output.shape[2] - bottom, left:output.shape[3] - right]

# Save and display results
cv2.imwrite("segmentation_output.png", output)

plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.title("Original Image")
plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

plt.subplot(1, 2, 2)
plt.title("Predicted Mask")
plt.imshow(output, cmap="gray")
plt.show()

print("Inference completed! Output saved as segmentation_output.png")
