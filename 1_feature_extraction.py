import cv2
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from skimage.morphology import skeletonize

#############
# Parametry #
#############

INPUT_DIR = "photos/"            # Domyślny katalog ze zdjęciami
SCALE_FACTOR = 0.25              # Współczynnik skalowania obrazu
FEATURES_LABELS = ["hull_area", "perimeter", "x", "y", "w", "h", "bounding_box_aspect_ratio",
                  "rect_area", "solidity", "equivalent_diameter", "ellipse_x_center",
                  "ellipse_y_center", "major_axis", "minor_axis", "ellipse_angle", "ellipse_ratio",
                  "contour_length", "skeleton_contour_length", "eccentricity", "circularity",
                  "elongation", "extent", "compactness", "form_factor", "centroid_x", "centroid_y",
                  "rectangularity", "solidity_ratio", "USER_ID"] # Etykiety dla cech

#####################
# Główna część kodu #
#####################

# Wybór katalogu ze zdjęciami do ekstrakcji cech
user_input = ""
while not os.path.isdir(user_input):
    print("Wpisz nazwę katalogu ze zdjęciami (domyślnie: 'photos'):")
    user_input = input("\n> ").strip()
    if not user_input:
        user_input = INPUT_DIR
    if not os.path.isdir(user_input):
        print(f"Błąd: Katalog '{user_input}' nie istnieje. Spróbuj ponownie.")
    else:
        INPUT_DIR = user_input

OUTPUT_DIR = "processed_" + INPUT_DIR
os.makedirs(OUTPUT_DIR, exist_ok=True)

file_list = sorted(os.listdir(INPUT_DIR))

feature_number = 1
data = []

if os.path.isdir(OUTPUT_DIR):
    processed_files = os.listdir(OUTPUT_DIR)
    if len(processed_files) == len(file_list):
        print("Wygląda na to, że cechy obrazów w podanym katalogu zostały już wyekstrahowane.")
        print(f"Jeśli chcesz ponownie wyekstrahować cechy, usuń zawartość katalogu 'processed_{INPUT_DIR}' i uruchom ponownie tę opcję.")
        exit(0)
    else:
        for i in range(0, len(file_list)):
            file_name = file_list[i]
            idUser = file_name.split("_")[0]
            print("Przetwarzanie obrazu " + file_name + "...")
            image = cv2.imread(os.path.join(INPUT_DIR, file_name))

            # Konwersja do skali szarości
            gray = cv2.resize(
                cv2.cvtColor(image, cv2.COLOR_BGR2GRAY),
                (0, 0),
                fx=SCALE_FACTOR,
                fy=SCALE_FACTOR,
                interpolation=cv2.INTER_AREA
            )

            gray = cv2.bilateralFilter(gray, d=11, sigmaColor=100, sigmaSpace=25)
            clahe = cv2.createCLAHE(clipLimit=3, tileGridSize=(8, 8))
            gray = clahe.apply(gray)

            # Binaryzacja
            binary = cv2.adaptiveThreshold(
                gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 15, 5
            )
            binary = 255 - cv2.medianBlur(binary, 5)

            # Skeletonizacja
            skeleton = skeletonize(255 - binary)
            skeleton = skeleton.astype(np.uint8) * 255
            output_image = 255 - skeleton
            convex_hull = skeleton

            cv2.imwrite(os.path.join(OUTPUT_DIR, file_name), output_image)

            contours, _ = cv2.findContours(
                255 - output_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )
            contour = max(contours, key=cv2.contourArea)

            contour_img = np.zeros_like(binary)
            cv2.drawContours(contour_img, contours, -1, 255, 1)
            cv2.imwrite(os.path.join(OUTPUT_DIR, file_name), contour_img)

            y, x = np.where(convex_hull == 255)
            points = np.column_stack((x, y)).astype(np.int32)
            hull = cv2.convexHull(points)

            dataTemp = []

            # 1. Pole powierzchni otoczki wypukłej (hull area)
            hull_area = cv2.contourArea(hull)
            dataTemp.append(hull_area)

            # 2. Długość konturu otoczki wypukłej (perimeter)
            perimeter = cv2.arcLength(hull, True)
            dataTemp.append(perimeter)

            # 3. Współrzędne prostokąta otaczającego (x, y, w, h)
            x, y, w, h = cv2.boundingRect(hull)
            dataTemp.append(x)
            dataTemp.append(y)
            dataTemp.append(w)
            dataTemp.append(h)

            # 4. Stosunek wysokości do szerokości prostokąta otaczającego (bounding_box_aspect_ratio)
            bounding_box_aspect_ratio = float(w) / h if h != 0 else 0
            dataTemp.append(bounding_box_aspect_ratio)

            # 5. Pole powierzchni prostokąta otaczającego (rect_area)
            rect_area = w * h
            dataTemp.append(rect_area)

            # 6. Zawartość otoczki wypukłej w prosotokącie otaczającym (solidity)
            solidity = float(rect_area) / hull_area
            dataTemp.append(solidity)

            # 7. Średnica równoważna (equivalent_diameter)
            equivalent_diameter = np.sqrt(4 * hull_area / np.pi)
            dataTemp.append(equivalent_diameter)

            # 8. Parametry elipsy otaczającej otoczkę wypukłą. 8a, 8b, 8c, 8d, 8e - środek, osie oraz kąt nachylenia elipsy (major_axis, minor_axis, angle, ellipse_ratio)
            ellipse = cv2.fitEllipse(hull)
            (x_center, y_center), (axis1, axis2), ellipse_angle = ellipse
            major_axis = max(axis1, axis2)
            minor_axis = min(axis1, axis2)
            ellipse_ratio = minor_axis / major_axis
            dataTemp.append(x_center)
            dataTemp.append(y_center)
            dataTemp.append(major_axis)
            dataTemp.append(minor_axis)
            dataTemp.append(ellipse_angle)
            dataTemp.append(ellipse_ratio)

            # 9. Długość konturu (contour_length)
            contour_length = len(contour)
            dataTemp.append(contour_length)

            # 10. Długość konturu szkieletu (skeleton_contour_length)
            skeleton_contour_length = len(skeleton)
            dataTemp.append(skeleton_contour_length)

            # 11. Mimośród elipsy (eccentricity)
            eccentricity = np.sqrt(1 - (minor_axis / major_axis) ** 2) if minor_axis != 0 else 0
            dataTemp.append(eccentricity)

            # 12. Krągłość (circularity)
            circularity = (4 * np.pi * hull_area) / (perimeter**2) if perimeter != 0 else 0
            dataTemp.append(circularity)

            # 13. Wydłużenie (elongation)
            elongation = major_axis / minor_axis if minor_axis != 0 else 0
            dataTemp.append(elongation)

            # 14. Stopień wypełnienia (extent)
            extent = hull_area / rect_area if rect_area != 0 else 0
            dataTemp.append(extent)

            # 15. Zwartość (compactness)
            compactness = (perimeter**2) / hull_area if hull_area != 0 else 0
            dataTemp.append(compactness)

            # 16. Współczynnik formy (form factor)
            form_factor = (4 * np.pi * hull_area) / (perimeter**2) if perimeter != 0 else 0
            dataTemp.append(form_factor)

            # 17a, 17b. Centroid
            M = cv2.moments(hull)
            if M["m00"] != 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
            else:
                cx, cy = 0, 0
            dataTemp.append(cx)
            dataTemp.append(cy)

            # 18. Prostokątność (rectangularity)
            rectangularity = hull_area / (w * h) if w * h != 0 else 0
            dataTemp.append(rectangularity)

            # 19. Współczynnik zawartość otoczki wypukłej w prostokącie otaczającym (solidity ratio)
            solidity_ratio = hull_area / rect_area if rect_area != 0 else 0
            dataTemp.append(solidity_ratio)

            # 20. Parametr decyzyjny - ID użytkownika (USER_ID)
            dataTemp.append(int(idUser))

            data.append(dataTemp)

persons = []
if data:
    max_id = max(int(row[-1]) for row in data)
    for user_id in range(1, max_id + 1):
        person_rows = [row for row in data if int(row[-1]) == user_id]
        persons.extend(person_rows)

persons = [[round(value, 3) for value in row] for row in persons]
columns = []
for i in range(len(FEATURES_LABELS)):
    columns.append(FEATURES_LABELS[i])

df = pd.DataFrame(persons, columns=columns)
df[df.columns[-1]] = df[df.columns[-1]].astype(int)
df.to_csv(f"data/{INPUT_DIR.strip('/')}_ear_data.csv", index=False)

#######################################################################
# Wyświetlenie w konsoli cech przykładowego obrazu w celu prezentacji #
#######################################################################

print(f"Pole powierzchni otoczki wypukłej:                                 {hull_area:.3f}")
print(f"Długość konturu otoczki wypukłej:                                  {perimeter:.3f}")
print(f"Współrzędne prostokąta otaczającego:                               x={x:.3f}, y={y:.3f}, w={w:.3f}, h={h:.3f}")
print(f"Stosunek wysokości do szerokości prostokąta otaczającego:          {bounding_box_aspect_ratio:.3f}")
print(f"Pole powierzchni prostokąta otaczającego:                          {rect_area:.3f}")
print(f"Zawartość otoczki wypukłej w prosotokącie otaczającym:             {solidity:.3f}")
print(f"Średnica równoważna:                                               {equivalent_diameter:.3f}")
print(f"Parametry elipsy otaczającej otoczkę wypukłą:                      x_center={x_center:.3f}, y_center={y_center:.3f}, major_axis={major_axis:.3f}, minor_axis={minor_axis:.3f}, angle={ellipse_angle:.3f}, ratio={ellipse_ratio:.3f}")
print(f"Długość konturu:                                                   {contour_length:.3f}")
print(f"Długość konturu szkieletu:                                         {skeleton_contour_length:.3f}")
print(f"Mimośród elipsy:                                                   {eccentricity:.3f}")
print(f"Krągłość:                                                          {circularity:.3f}")
print(f"Wydłużenie:                                                        {elongation:.3f}")
print(f"Stopień wypełnienia:                                               {extent:.3f}")
print(f"Zwartość:                                                          {compactness:.3f}")
print(f"Współczynnik formy:                                                {form_factor:.3f}")
print(f"Centroid:                                                          x={cx:.3f}, y={cy:.3f}")
print(f"Prostokątność:                                                     {rectangularity:.3f}")
print(f"Współczynnik zawartość otoczki wypukłej w prostokącie otaczającym: {solidity_ratio:.3f}")
print(f"ID użytkownika:                                                    {idUser}")