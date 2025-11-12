import cv2
import os
import numpy as np
import mahotas
import pandas as pd
import matplotlib.pyplot as plt
from fontTools.misc.textTools import tostr

from skimage.data import page
from skimage.filters import threshold_otsu, threshold_niblack, threshold_sauvola
from skimage.morphology import skeletonize

PERCENT_SCALE = 50
ROTATE_ANGLE = 25

def rotate_image(image, angle):
    (h, w) = image.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    return cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101)

def add_gaussian_noise(image, std_dev):
    """Add Gaussian noise with given standard deviation."""
    row, col, ch = image.shape
    gauss = np.random.normal(0, std_dev * 255, (row, col, ch)).astype('float32')
    noisy = image.astype('float32') + gauss
    noisy = np.clip(noisy, 0, 255).astype('uint8')
    return noisy

def change_brightness(image, factor):
    """Adjust brightness by a certain factor."""
    pil_img = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    enhancer = ImageEnhance.Brightness(pil_img)
    bright_img = enhancer.enhance(factor)
    return cv2.cvtColor(np.array(bright_img), cv2.COLOR_RGB2BGR)

def augment_data(person, target_samples=10, noise_factor=0.05):
    augmented_data = person.copy()

    np.random.seed(0)

    while len(augmented_data) < target_samples:
        # Wybierz losowo dwie próbki do interpolacji
        idx1, idx2 = np.random.choice(len(person), 2, replace=False)
        sample1, sample2 = np.array(augmented_data[idx1]), np.array(augmented_data[idx2])
        # Interpolacja liniowa
        alpha = np.random.rand()
        new_sample = alpha * sample1 + (1 - alpha) * sample2
        # Dodanie szumu
        noise = np.random.randn(*new_sample.shape) * noise_factor * np.std(augmented_data, axis=0)
        new_sample = new_sample + noise
        # Konwersja na listę i dodanie do danych
        augmented_data.append(new_sample.tolist())

    return augmented_data[:target_samples]

# Wczytanie obrazu i pobranie jego nazwy
image_path = 'Photos/'
file_list = os.listdir(image_path)

feature_number = 1
data = []

for i in range(0, len(file_list)):
    file_name = file_list[i]
    idUser = file_name.split("_")[1]
    print(idUser, ' ', file_name, ' ', ' Preparing ...')
    image = cv2.imread(image_path + file_name)

    scale_factor = 0.25

    gray = cv2.resize(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY), (0, 0), fx=scale_factor, fy=scale_factor, interpolation=cv2.INTER_AREA)

    gray = rotate_image(gray, ROTATE_ANGLE)

    #################
    # Binarization. #
    #################

    # Adaptive method
    binary = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 15, 5)
    binary = 255 - cv2.medianBlur(binary, 5)

    # === Hu and Zernike moments (dla obrazu binarnego) ===
    hu_moments = cv2.HuMoments(cv2.moments(binary)).flatten()

    radius = min(binary.shape) // 2
    center = (binary.shape[1] // 2, binary.shape[0] // 2)
    zernike_patch = binary[
        center[1] - radius : center[1] + radius,
        center[0] - radius : center[0] + radius
    ]
    zernike_moments = mahotas.features.zernike_moments(zernike_patch, radius)

    data.append(list(zernike_moments) + list(hu_moments) + [int(idUser)])

# Augumentation.
maxId = int(idUser)
persons = []
for j in range(1, maxId + 1):
    person = []
    for i in range(0, len(data)):
        if (data[i][len(data[i])-1]) == j:
            person.append(data[i])
    person_aug = augment_data(person, target_samples=10)
    for record in person_aug:
        persons.append(record)

print("Augumentation!")

persons = [[round(value, 3) for value in row] for row in persons]

columns = []
for i in range(0, len(persons[0])):
    columns.append("f" + str(i))

df = pd.DataFrame(persons, columns=columns)
df[df.columns[-1]] = df[df.columns[-1]].astype(int)
df.to_csv('Data/ear_data_aug_Zernike_Hu_rot'+str(ROTATE_ANGLE)+'.csv', index=False)
