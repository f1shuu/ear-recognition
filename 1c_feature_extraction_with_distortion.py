import cv2
import os
import numpy as np
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
    """Add Gaussian noise with given standard deviation (grayscale or color)."""
    if len(image.shape) == 2:  # Grayscale
        row, col = image.shape
        gauss = np.random.normal(0, std_dev * 255, (row, col)).astype('float32')
    else:  # Color
        row, col, ch = image.shape
        gauss = np.random.normal(0, std_dev * 255, (row, col, ch)).astype('float32')

    noisy = image.astype('float32') + gauss
    noisy = np.clip(noisy, 0, 255).astype('uint8')

    return noisy

def change_brightness(image, factor):
    """Adjust brightness by a certain factor (grayscale or color)."""
    if len(image.shape) == 2:  # Grayscale
        bright_img = image.astype('float32') * factor
        bright_img = np.clip(bright_img, 0, 255).astype('uint8')
    else:  # Color
        pil_img = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        enhancer = ImageEnhance.Brightness(pil_img)
        bright_img = enhancer.enhance(factor)
        bright_img = cv2.cvtColor(np.array(bright_img), cv2.COLOR_RGB2BGR)
    return bright_img

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

def filter_skeleton(skeleton, size):
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(skeleton)

    filtered_skeleton = np.zeros_like(skeleton)
    for i in range(1, num_labels):
        if stats[i, cv2.CC_STAT_AREA] >= size:
            filtered_skeleton[labels == i] = 255

    return filtered_skeleton

# Wczytanie obrazu i pobranie jego nazwy
image_path = 'Photos/'
file_list = os.listdir(image_path)

feature_number = 1
data = []

for i in range(2, len(file_list)):
    file_name = file_list[i]
    idUser = file_name.split("_")[1]
    print(idUser, ' ', file_name, ' ', ' Preparing ...')
    image = cv2.imread(image_path + file_name)

    scale_factor = 0.25
    gray = cv2.resize(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY), (0, 0), fx=scale_factor, fy=scale_factor, interpolation=cv2.INTER_AREA)

    # Lekki szum	    0.025   Ledwo zauważalny szum
    # Umiarkowany szum	0.050   Widoczny szum, realistyczny
    # Silny szum	    0.075   Silny szum, trudny przypadek
    # Bardzo silny	    0.100   Skrajny przypadek, test odporności
    #gray = add_gaussian_noise(gray, 0.1)

    # Przyciemnienie	    0.50    Wyraźnie ciemniejszy obraz
    # Lekko ciemniejszy	    0.75    Naturalne przyciemnienie
    # Lekko jaśniejszy	    1.25	Jasność +25%
    # Silne rozjaśnienie	1.50	Jasność +50%
    gray = change_brightness(gray, 1.50)

    #####################
    # Image processing. #
    #####################

    gray = cv2.bilateralFilter(gray, d=11, sigmaColor=100, sigmaSpace=25)
    clahe = cv2.createCLAHE(clipLimit=3, tileGridSize=(8, 8))
    gray = clahe.apply(gray)

    # Binarization. Adaptive method
    binary = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 15, 5)
    binary = 255 - cv2.medianBlur(binary, 1)

    # Skeletonization.
    #skeleton = skeletonize(255 - binary)
    #skeleton = skeleton.astype(np.uint8) * 255
    #output_image = 255 - skeleton
    #convex_hull = skeleton

    skeleton = skeletonize(255 - binary)
    skeleton = skeleton.astype(np.uint8) * 255
    skeleton = filter_skeleton(skeleton, 20)
    output_image = 255 - skeleton
    convex_hull = skeleton

    #cv2.imshow("Binary", skeleton)
    #cv2.waitKey(0)
    #cv2.destroyAllWindows()
    #exit()

    y, x = np.where(convex_hull == 255)
    points = np.column_stack((x, y)).astype(np.int32)
    hull = cv2.convexHull(points)

    dataTemp = []

    # Cechy pomocnicze.
    # Pole powierzchni otoczki wypukłej.
    hull_area = cv2.contourArea(hull)
    # Długość konturu otoczki wypukłej.
    perimeter = cv2.arcLength(hull, True)
    # Współrzędne prostokąta otaczającego.
    x, y, w, h = cv2.boundingRect(hull)
    # Pole powierzchni prostokąta otaczającego.
    rect_area = w * h

    ###################################################################################################################
    # 1. Aspect ratio - stosunek wysokości do szerokości prostokąta otaczającego.
    bounding_box_aspect_ratio = float(w) / h if h != 0 else 0
    dataTemp.append(bounding_box_aspect_ratio)

    # 2. Zawartość otoczki wypukłej w prosotokącie otaczającym.
    solidity = float(rect_area) / hull_area
    dataTemp.append(solidity)

    # 3 i 4. Parametry elipsy otaczającej otoczkę wypukłą.
    ellipse = cv2.fitEllipse(hull)
    (x_center, y_center), (axis1, axis2), angle = ellipse
    major_axis = max(axis1, axis2)
    minor_axis = min(axis1, axis2)
    dataTemp.append(major_axis)
    dataTemp.append(minor_axis)

    # 5. Eccentricity (Mimośród elipsy).
    eccentricity = np.sqrt(1 - (minor_axis / major_axis) ** 2) if minor_axis != 0 else 0
    dataTemp.append(eccentricity)

    # 6. Circularity (krągłość).
    circularity = (4 * np.pi * hull_area) / (perimeter ** 2) if perimeter != 0 else 0
    dataTemp.append(circularity)

    # 7. Elongation (wydłużenie)
    elongation = major_axis / minor_axis if minor_axis != 0 else 0
    dataTemp.append(elongation)

    # 8, 9. Centroid cx, cy !!!!!
    M = cv2.moments(hull)
    if M["m00"] != 0:
        cx = int(M["m10"] / M["m00"])
        cy = int(M["m01"] / M["m00"])
    else:
        cx, cy = 0, 0
    dataTemp.append(cx)
    dataTemp.append(cy)

    # 10. Rectangularity.
    rectangularity = hull_area / (w * h) if w * h != 0 else 0
    dataTemp.append(rectangularity)

    # 11. Solidity ratio.
    solidity_ratio = hull_area / rect_area if rect_area != 0 else 0
    dataTemp.append(solidity_ratio)

    dataTemp.append(int(idUser))
    data.append(dataTemp)

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
df.to_csv('Data/ear_data_aug_denoise_bright_light150.csv', index=False)
