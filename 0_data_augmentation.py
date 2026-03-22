import cv2
import os
import numpy as np
from PIL import Image, ImageEnhance

################################
# Parametry augmentacji danych #
################################

INPUT_DIR = "photos/"  # Katalog z oryginalnymi zdjęciami
OUTPUT_DIR = "photos/"  # Katalog docelowy dla przetworzonych zdjęć (domyślnie ten sam)
ROTATE_ANGLE = 25  # Kąt obrotu obrazu w stopniach
NOISE_STD_DEV = 0.5  # Odchylenie standardowe szumu Gaussa
BRIGHTNESS_FACTOR = 0.75  # Współczynnik zmiany jasności

#################################
# Funkcje do augmentacji danych #
#################################


# 1. Obrót obrazu o określony kąt
def rotate_image(image, angle):
    (h, w) = image.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    return cv2.warpAffine(
        image, M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101
    )


# 2. Dodanie szumu Gaussa z podaną wartością odchylenia standardowego
def add_gaussian_noise(image, std_dev):
    if len(image.shape) == 2:  # Grayscale
        row, col = image.shape
        gauss = np.random.normal(0, std_dev * 255, (row, col)).astype("float32")
    else:  # Color
        row, col, ch = image.shape
        gauss = np.random.normal(0, std_dev * 255, (row, col, ch)).astype("float32")

    noisy = image.astype("float32") + gauss
    noisy = np.clip(noisy, 0, 255).astype("uint8")

    return noisy


# 3. Zmiana jasności obrazu przez określony współczynnik
def change_brightness(image, factor):
    if len(image.shape) == 2:  # Grayscale
        bright_img = image.astype("float32") * factor
        bright_img = np.clip(bright_img, 0, 255).astype("uint8")
    else:  # Color
        pil_img = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        enhancer = ImageEnhance.Brightness(pil_img)
        bright_img = enhancer.enhance(factor)
        bright_img = cv2.cvtColor(np.array(bright_img), cv2.COLOR_RGB2BGR)
    return bright_img


################################
# Początek głównej części kodu #
################################

file_list_before = sorted(os.listdir(INPUT_DIR))

print("Rozpoczynam augmentację danych...")

for file in file_list_before:
    file_name = os.path.splitext(file)[0]
    if file_name[-1:] == "0":
        file_name = file_name[:-2]
        image = cv2.imread(os.path.join(INPUT_DIR, file))

        # 1. Obrót obrazu o określony kąt
        if not os.path.exists(os.path.join(OUTPUT_DIR, f"{file_name}_1.jpg")):
            rotated_image = rotate_image(image, ROTATE_ANGLE)
            print("Tworzenie obrazu " + file_name + "_1.jpg...")
            cv2.imwrite(os.path.join(OUTPUT_DIR, f"{file_name}_1.jpg"), rotated_image)
        else:
            print("Obraz " + file_name + "_1.jpg już istnieje, pomijanie...")

        # 2. Dodanie szumu Gaussa z podaną wartością odchylenia standardowego
        if not os.path.exists(os.path.join(OUTPUT_DIR, f"{file_name}_2.jpg")):
            noisy_image = add_gaussian_noise(image, NOISE_STD_DEV)
            print("Tworzenie obrazu " + file_name + "_2.jpg...")
            cv2.imwrite(os.path.join(OUTPUT_DIR, f"{file_name}_2.jpg"), noisy_image)
        else:
            print("Obraz " + file_name + "_2.jpg już istnieje, pomijanie...")

        # 3. Zmiana jasności obrazu przez określony współczynnik
        if not os.path.exists(os.path.join(OUTPUT_DIR, f"{file_name}_3.jpg")):
            brightened_image = change_brightness(image, BRIGHTNESS_FACTOR)
            print("Tworzenie obrazu " + file_name + "_3.jpg...")
            cv2.imwrite(
                os.path.join(OUTPUT_DIR, f"{file_name}_3.jpg"), brightened_image
            )
        else:
            print("Obraz " + file_name + "_3.jpg już istnieje, pomijanie...")

        # 4. Połączenie wszystkich transformacji
        if not os.path.exists(os.path.join(OUTPUT_DIR, f"{file_name}_4.jpg")):
            combined_image = change_brightness(
                add_gaussian_noise(rotate_image(image, ROTATE_ANGLE), NOISE_STD_DEV),
                BRIGHTNESS_FACTOR,
            )
            print("Tworzenie obrazu " + file_name + "_4.jpg...")
            cv2.imwrite(os.path.join(OUTPUT_DIR, f"{file_name}_4.jpg"), combined_image)
        else:
            print("Obraz " + file_name + "_4.jpg już istnieje, pomijanie...")

file_list_after = sorted(os.listdir(OUTPUT_DIR))

print("Augmentacja zakończona.")
print("Liczba obrazów przed augmentacją: " + str(len(file_list_before)))
print("Liczba obrazów po augmentacji: " + str(len(file_list_after)))
