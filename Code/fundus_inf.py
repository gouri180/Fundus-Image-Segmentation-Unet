import torch
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
import cv2
import numpy as np


import torch
import torch.nn as nn

class conv_block(nn.Module):
    def __init__(self, in_c, out_c):  # FIXED __init__
        super(conv_block, self).__init__()

        self.conv1 = nn.Conv2d(in_c, out_c, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_c)

        self.conv2 = nn.Conv2d(out_c, out_c, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_c)

        self.relu = nn.ReLU(inplace=True)

    def forward(self, inputs):
        x = self.conv1(inputs)
        x = self.bn1(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)

        return x


class encoder_block(nn.Module):
    def __init__(self, in_c, out_c):  # FIXED __init__
        super(encoder_block, self).__init__()

        self.conv = conv_block(in_c, out_c)
        self.pool = nn.MaxPool2d((2, 2))

    def forward(self, inputs):
        x = self.conv(inputs)
        p = self.pool(x)

        return x, p


class decoder_block(nn.Module):
    def __init__(self, in_c, out_c):  # FIXED __init__
        super(decoder_block, self).__init__()

        self.t_conv = nn.ConvTranspose2d(in_c, out_c, kernel_size=2, stride=2, padding=0)
        self.conv = conv_block(out_c + out_c, out_c)

    def forward(self, inputs, skip):
        x = self.t_conv(inputs)
        x = torch.cat([x, skip], axis=1)
        x = self.conv(x)

        return x


class Unet(nn.Module):
    def __init__(self):  # FIXED __init__
        super(Unet, self).__init__()

        """Encoder"""
        self.e1 = encoder_block(3, 64)
        self.e2 = encoder_block(64, 128)
        self.e3 = encoder_block(128, 256)
        self.e4 = encoder_block(256, 512)

        """Bottle-neck"""
        self.b = conv_block(512, 1024)

        """Decoder"""
        self.d1 = decoder_block(1024, 512)
        self.d2 = decoder_block(512, 256)
        self.d3 = decoder_block(256, 128)
        self.d4 = decoder_block(128, 64)

        """Classification Layer"""
        self.output = nn.Conv2d(64, 1, kernel_size=1, padding=0)

    def forward(self, inputs):
        e1, p1 = self.e1(inputs)
        e2, p2 = self.e2(p1)
        e3, p3 = self.e3(p2)
        e4, p4 = self.e4(p3)

        b = self.b(p4)

        d1 = self.d1(b, e4)
        d2 = self.d2(d1, e3)
        d3 = self.d3(d2, e2)
        d4 = self.d4(d3, e1)

        output = self.output(d4)

        return output

# Load Model
model = Unet()
model.load_state_dict(torch.load(r"C:\Users\iamgo\Downloads\fundus_inf\Models\fundus_segmentation_2final_model (1).pth", map_location=torch.device('cpu')))
model.eval()  # Set to evaluation mode

# model.load_state_dict(torch.load(r'C:\Users\ABINSHA IBNU AYYOOB\PycharmProjects\language_detection\fundus_segmentation_2final_model (1).pth'))  # load weights to model.
# model.to(device)




# need to pad the images which are of different size.(Padding images to a consistent size helps maintain their original aspect ratio without distortion.)

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


model.eval()

image_path = r"C:\Users\iamgo\Downloads\fundus_inf\Fundus_Data\i vision eye hospital\GLAUCOMA\2021 7267_20231227140458205.jpg"  # Update this path
image = cv2.imread(image_path, cv2.IMREAD_COLOR)
#print(image.shape)

if image is None:
    print('image is none')
else:
    print(f"Original image size: {image.shape}")
    padded_image, padding = pad_to_divisibility(image)
    print(f"Padded image size: {padded_image.shape}")

    # x = cv2.resize(image, (512,512))
    x = np.transpose(padded_image, (2, 0, 1))

    x = x / 255.0
    x = np.expand_dims(x, axis=0)
    x = x.astype(np.float32)
    x = torch.from_numpy(x)
    #x = x.to(device)

with torch.no_grad():
    output = model(x)

output = torch.sigmoid(output)
output = (output > 0.3).float()  # adjust the threshold to get accurate output
output = output.cpu().numpy()[0][0]

top, bottom, left, right = padding
output = output[top:output.shape[0] - bottom, left:output.shape[1] - right]

plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.title("Original Image")
plt.imshow(image)

plt.subplot(1, 2, 2)
plt.title("Predicted Mask")
plt.imshow(output, cmap='gray')
plt.show()