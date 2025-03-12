import os
import numpy as np
import cv2
from glob import glob
from tqdm import tqdm
import imageio
from albumentations import HorizontalFlip, VerticalFlip, Rotate

""" Create directories to save the augmented data """
os.makedirs("/kaggle/working/train/image/")
os.makedirs("/kaggle/working/train/mask/")
os.makedirs("/kaggle/working/test/image/")
os.makedirs("/kaggle/working/test/mask/")

path = '/kaggle/input/drive-final/drive_fundus'   #upload the zip into the dataset in Kaggle
train_x = sorted(glob(os.path.join(path, "training", "images", "*.tif")))
train_y = sorted(glob(os.path.join(path, "training", "1st_manual", "*.gif")))

test_x = sorted(glob(os.path.join(path, "test", "images", "*.tif")))
test_y = sorted(glob(os.path.join(path, "test", "1st_manual", "*.gif")))




np.random.seed(42)


def augment_data(images, masks, save_path, augment=True):
    size = (512, 512)

    for idx, (x, y) in tqdm(enumerate(zip(images, masks)), total=len(images)):
        """ Extracting the name """
        name = x.split("/")[-1].split(".")[0]

        """ Reading image and mask """
        x = cv2.imread(x, cv2.IMREAD_COLOR)
        y = imageio.mimread(y)[0]

        if augment == True:
            aug = HorizontalFlip(p=1.0)
            augmented = aug(image=x, mask=y)
            x1 = augmented["image"]
            y1 = augmented["mask"]

            aug = VerticalFlip(p=1.0)
            augmented = aug(image=x, mask=y)
            x2 = augmented["image"]
            y2 = augmented["mask"]

            aug = Rotate(limit=45, p=1.0)
            augmented = aug(image=x, mask=y)
            x3 = augmented["image"]
            y3 = augmented["mask"]

            aug = RandomCrop(height=400, width=400, p=0.5)
            augmented = aug(image=x, mask=y)
            x4 = augmented["image"]
            y4 = augmented["mask"]

            X = [x, x1, x2, x3, x4]
            Y = [y, y1, y2, y3, y4]

        else:
            X = [x]
            Y = [y]

        index = 0
        for i, m in zip(X, Y):
            i = cv2.resize(i, size)
            m = cv2.resize(m, size)

            tmp_image_name = f"{name}_{index}.png"
            tmp_mask_name = f"{name}_{index}.png"

            image_path = os.path.join(save_path, "image", tmp_image_name)
            mask_path = os.path.join(save_path, "mask", tmp_mask_name)

            cv2.imwrite(image_path, i)
            cv2.imwrite(mask_path, m)

            index += 1



    

    """ Data augmentation """
augment_data(train_x, train_y, "/kaggle/working/train", augment=True)
augment_data(test_x, test_y, "/kaggle/working/test", augment=False)


