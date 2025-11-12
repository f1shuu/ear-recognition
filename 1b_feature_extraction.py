import cv2
import os
import numpy as np
import pandas as pd
import time
import matplotlib.pyplot as plt
from fontTools.misc.textTools import tostr

from skimage.data import page
from skimage.filters import threshold_otsu, threshold_niblack, threshold_sauvola
from skimage.morphology import skeletonize

PERCENT_SCALE = 50

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

start = time.time()

for i in range(0, len(file_list)):
    file_name = file_list[i]
    idUser = file_name.split("_")[1]
    print(idUser, ' ', file_name, ' ', ' Preparing ...')
    image = cv2.imread(image_path + file_name)

    scale_factor = 0.25
    gray = cv2.resize(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY), (0, 0), fx=scale_factor, fy=scale_factor, interpolation=cv2.INTER_AREA)

    #################
    # Binarization. #
    #################

    # Adaptive method
    binary = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 15, 5)
    #binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8), iterations=3)
    binary = 255 - cv2.medianBlur(binary, 5)

    #cv2.imwrite(file_name, binary)

    # Otsu.
    #threshold_otsu = threshold_otsu(gray)
    #binary = 255 - (gray > threshold_otsu).astype(np.uint8) * 255

    # Niblack.
    #window_size = 25
    #thresh_niblack = threshold_niblack(gray, window_size=window_size, k=0.1)
    #binary = 255 - (gray > thresh_niblack).astype(np.uint8) * 255

    # Sauvola.
    #window_size = 25
    #thresh_sauvola = threshold_sauvola(gray, window_size=window_size)
    #binary = 255 - (gray > thresh_sauvola).astype(np.uint8) * 255

    ########################################################################################################################

    skeleton = skeletonize(255 - binary)
    skeleton = skeleton.astype(np.uint8) * 255
    output_image = 255 - skeleton
    convex_hull = skeleton

    #cv2.imwrite(file_name, output_image)

    #contours, _ = cv2.findContours(255-output_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    #contour = max(contours, key=cv2.contourArea)

    #contour_img = np.zeros_like(binary)
    #cv2.drawContours(contour_img, contours, -1, 255, 1)
    #cv2.imwrite(file_name, contour_img)

    y, x = np.where(convex_hull == 255)
    points = np.column_stack((x, y)).astype(np.int32)
    hull = cv2.convexHull(points)

    dataTemp = []

    # 1. Pole powierzchni otoczki wypukłej.
    hull_area = cv2.contourArea(hull)
    #dataTemp.append(hull_area)

    # 2. Długość konturu otoczki wypukłej.
    perimeter = cv2.arcLength(hull, True)
    #dataTemp.append(perimeter)

    # 3. Współrzędne prostokąta otaczającego.
    x, y, w, h = cv2.boundingRect(hull)
    #dataTemp.append(x)
    #dataTemp.append(y)
    #dataTemp.append(w)
    #dataTemp.append(h)

    # 4. Aspect ratio - stosunek wysokości do szerokości prostokąta otaczającego.
    # 1. !!!!!!
    bounding_box_aspect_ratio = float(w) / h if h != 0 else 0
    dataTemp.append(bounding_box_aspect_ratio)

    # 5 .Pole powierzchni prostokąta otaczającego.
    rect_area = w * h
    #dataTemp.append(rect_area)

    # 6. Zawartość otoczki wypukłej w prosotokącie otaczającym.
    # 2. !!!!!!
    solidity = float(rect_area) / hull_area
    dataTemp.append(solidity)

    # 7. Średnica równoważna.
    #equivalent_diameter = np.sqrt(4 * hull_area / np.pi)
    #dataTemp.append(equivalent_diameter)

    # 8. Parametry elipsy otaczającej otoczkę wypukłą.
    # 8a, 8b, 8c, 8d, 8e - środek, osie oraz kąt nachylenia elipsy.
    # 3 i 4. !!!!!!
    ellipse = cv2.fitEllipse(hull)
    (x_center, y_center), (axis1, axis2), angle = ellipse
    major_axis = max(axis1, axis2)
    minor_axis = min(axis1, axis2)
    #ellipse_ratio = minor_axis / major_axis
    #dataTemp.append(x_center)
    #dataTemp.append(y_center)
    dataTemp.append(major_axis)
    dataTemp.append(minor_axis)
    #dataTemp.append(angle)
    #dataTemp.append(ellipse_ratio)

    # 9. Długość konturu.
    #contour_length = len(contour)
    #dataTemp.append(contour_length)

    # 10. Długość konturu (szkieletu).
    #skeleton_contour_length = len(skeleton)
    #dataTemp.append(skeleton_contour_length)

    # 11. Eccentricity (Mimośród elipsy).
    # 5. !!!!
    eccentricity = np.sqrt(1 - (minor_axis / major_axis) ** 2) if minor_axis != 0 else 0
    dataTemp.append(eccentricity)

    # 12. Circularity (krągłość).
    # 6. !!!!
    circularity = (4 * np.pi * hull_area) / (perimeter ** 2) if perimeter != 0 else 0
    dataTemp.append(circularity)

    # 13. Elongation (wydłużenie)
    # 7. !!!!
    elongation = major_axis / minor_axis if minor_axis != 0 else 0
    dataTemp.append(elongation)

    # 14. Extent (Stopień wypełnienia).
    #extent = hull_area / rect_area if rect_area != 0 else 0
    #dataTemp.append(extent)

    # 15. Compactness (Zwartość).
    #compactness = (perimeter ** 2) / hull_area if hull_area != 0 else 0
    #dataTemp.append(compactness)

    # 16. Form Factor (Współczynnik Formy).
    #form_factor = (4 * np.pi * hull_area) / (perimeter ** 2) if perimeter != 0 else 0
    #dataTemp.append(form_factor)

    # 17a, 17b. Centroid.
    # 8, 9 !!!!!
    M = cv2.moments(hull)
    if M["m00"] != 0:
        cx = int(M["m10"] / M["m00"])
        cy = int(M["m01"] / M["m00"])
    else:
        cx, cy = 0, 0
    dataTemp.append(cx)
    dataTemp.append(cy)

    # 18. Rectangularity.
    # 10 !!!!
    rectangularity = hull_area / (w * h) if w * h != 0 else 0
    dataTemp.append(rectangularity)

    # 19. Solidity ratio.
    # 11 !!!!!
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

end = time.time()
print(f"Czas wykonania: {end - start:.4f} sekund")

persons = [[round(value, 3) for value in row] for row in persons]

columns = []
for i in range(0, len(persons[0])):
    columns.append("f" + str(i))

df = pd.DataFrame(persons, columns=columns)
df[df.columns[-1]] = df[df.columns[-1]].astype(int)
df.to_csv('Data/ear_data_aug_time.csv', index=False)

# Wyświetlenie efektu końcowego
'''
plt.figure(figsize=(10, 10))
plt.imshow(binary,'gray')
plt.axis('off')
plt.show()
'''

'''
# Features.


# 3. Współrzędne prostokąta otaczającego.
x, y, w, h = cv2.boundingRect(hull)

# 4. Aspect ratio - stosunek wysokości do szerokości prostokąta otaczającego.
bounding_box_aspect_ratio = float(w) / h if h != 0 else 0

# 5 .Pole powierzchni prostokąta otaczającego.
rect_area = w*h

# 6. Zawartość otoczki wypukłej w prosotokącie otaczającym.
solidity = float(rect_area) / hull_area

# 7. Średnica równoważna.
equivalent_diameter = np.sqrt(4 * hull_area / np.pi)

# 8. Parametry elipsy otaczającej otoczkę wypukłą.
# 8a, 8b, 8c, 8d, 8e - środek, osie oraz kąt nachylenia elipsy.
ellipse = cv2.fitEllipse(hull)
(x_center, y_center), (axis1, axis2), angle = ellipse
major_axis = max(axis1, axis2)
minor_axis = min(axis1, axis2)
ellipse_ratio = minor_axis / major_axis

# 9. Długość konturu.
contour_length = len(contour)

# 10. Długość konturu (szkieletu).
skeleton_contour_length = len(skeleton)

# 11. Eccentricity (Mimośród elipsy).
eccentricity = np.sqrt(1 - (minor_axis / major_axis) ** 2) if minor_axis != 0 else 0

# 12. Circularity (krągłość).
circularity = (4 * np.pi * hull_area) / (perimeter ** 2) if perimeter != 0 else 0

# 13. Elongation (wydłużenie)
elongation = major_axis / minor_axis if minor_axis != 0 else 0

# 14. Extent (Stopień wypełnienia).
extent = hull_area / rect_area if rect_area != 0 else 0

# 15. Compactness (Zwartość).
compactness = (perimeter ** 2) / hull_area if hull_area != 0 else 0

# 16. Form Factor (Współczynnik Formy).
form_factor = (4 * np.pi * hull_area) / (perimeter ** 2) if perimeter != 0 else 0

# 17a, 17b. Centroid.
M = cv2.moments(hull)
if M["m00"] != 0:
    cx = int(M["m10"] / M["m00"])
    cy = int(M["m01"] / M["m00"])
else:
    cx, cy = 0, 0

# 18. Rectangularity.
rectangularity = hull_area / (w * h) if w * h != 0 else 0

# 19. Solidity ratio.
solidity_ratio = hull_area / rect_area if rect_area != 0 else 0

print(f"Hull area: {hull_area:.3f}")
print(f"Rectangular Area: {rect_area:.3f}")

print(f"Perimeter: {perimeter:.3f}")

print(f"Width: {w:.3f}")
print(f"Height: {h:.3f}")
print(f"Bounding Box Aspect Ratio: {bounding_box_aspect_ratio:.3f}")

print(f"Solidity: {solidity:.3f}")
print(f"Equivalent Diameter: {equivalent_diameter:.3f}")

print(f"Ellipse Major Axis: {major_axis:.3f}")
print(f"Ellipse Minor Axis: {minor_axis:.3f}")
print(f"Ellipse Ratio: {ellipse_ratio:.3f}")
print(f"Ellipse Angle: {angle:.3f}")

print(f"Contour Length: {contour_length:.3f}")
print(f"Skeleton contour Length: {skeleton_contour_length:.3f}")

print(f"Eccentricity: {eccentricity:.3f}")
print(f"Circularity: {circularity:.3f}")
print(f"Elongation: {elongation:.3f}")
print(f"Extent: {extent:.3f}")
print(f"Compactness: {compactness:.3f}")

print(f"Form Factor: {form_factor:.3f}")
print(f"Centroid X: {cx:.3f}")
print(f"Centroid Y: {cy:.3f}")

print(f"Rectangularity: {rectangularity:.3f}")
print(f"Solidity Ratio: {solidity_ratio:.3f}")

# 20. Momenty Hu.
moments = cv2.moments(hull)
hu_moments = cv2.HuMoments(moments).flatten()
for i, moment in enumerate(hu_moments):
    print(f"Hu[{i+1}]: {moment:.3f}")
'''
