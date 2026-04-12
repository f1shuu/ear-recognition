import cv2
import os
import numpy as np
from PIL import Image, ImageEnhance

#############
# Parametry #
#############

INPUT_DIR = "photos/"    # Domyślny katalog ze zdjęciami
ROTATE_ANGLE = 25        # Kąt obrotu obrazu w stopniach
NOISE_STD_DEV = 0.5      # Odchylenie standardowe szumu Gaussa
BRIGHTNESS_FACTOR = 0.75 # Współczynnik zmiany jasności

######################
# Funkcje pomocnicze #
######################

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

#####################
# Główna część kodu #
#####################

# Wybór katalogu ze zdjęciami do augmentacji
user_input = ""
while not os.path.isdir(user_input):
    print("Wpisz nazwę katalogu z oryginalnymi zdjęciami (domyślnie: 'photos'):")
    user_input = input("\n> ").strip()
    if not user_input:
        user_input = INPUT_DIR
    if not os.path.isdir(user_input):
        print(f"Błąd: Katalog '{user_input}' nie istnieje. Spróbuj ponownie.")
    else:
        INPUT_DIR = user_input

print("Rozpoczynam augmentację danych...")
file_list_before = sorted(os.listdir(INPUT_DIR))

for file in file_list_before:
    file_name = os.path.splitext(file)[0]
    file_ext = os.path.splitext(file)[1].lower()
    if file_name[-1:] == "0":
        file_name = file_name[:-2]
        image = cv2.imread(os.path.join(INPUT_DIR, file))

        # 1. Obrót obrazu o określony kąt
        if not os.path.exists(os.path.join(INPUT_DIR, f"{file_name}_1{file_ext}")):
            rotated_image = rotate_image(image, ROTATE_ANGLE)
            print("Tworzenie obrazu " + file_name + "_1" + file_ext + "...")
            cv2.imwrite(os.path.join(INPUT_DIR, f"{file_name}_1{file_ext}"), rotated_image)

        # 2. Dodanie szumu Gaussa z podaną wartością odchylenia standardowego
        if not os.path.exists(os.path.join(INPUT_DIR, f"{file_name}_2{file_ext}")):
            noisy_image = add_gaussian_noise(image, NOISE_STD_DEV)
            print("Tworzenie obrazu " + file_name + "_2" + file_ext + "...")
            cv2.imwrite(os.path.join(INPUT_DIR, f"{file_name}_2{file_ext}"), noisy_image)

        # 3. Zmiana jasności obrazu przez określony współczynnik
        if not os.path.exists(os.path.join(INPUT_DIR, f"{file_name}_3{file_ext}")):
            brightened_image = change_brightness(image, BRIGHTNESS_FACTOR)
            print("Tworzenie obrazu " + file_name + "_3" + file_ext + "...")
            cv2.imwrite(
                os.path.join(INPUT_DIR, f"{file_name}_3{file_ext}"), brightened_image
            )

        # 4. Połączenie wszystkich transformacji
        if not os.path.exists(os.path.join(INPUT_DIR, f"{file_name}_4{file_ext}")):
            combined_image = change_brightness(
                add_gaussian_noise(rotate_image(image, ROTATE_ANGLE), NOISE_STD_DEV),
                BRIGHTNESS_FACTOR,
            )
            print("Tworzenie obrazu " + file_name + "_4" + file_ext + "...")
            cv2.imwrite(os.path.join(INPUT_DIR, f"{file_name}_4{file_ext}"), combined_image)

file_list_after = sorted(os.listdir(INPUT_DIR))

print("\nAugmentacja zakończona.")
print("Liczba obrazów przed augmentacją: " + str(len(file_list_before)))
print("Liczba obrazów po augmentacji:    " + str(len(file_list_after)) + " (+" + str(len(file_list_after) - len(file_list_before)) + ")")
if (len(file_list_before) * 5) != len(file_list_after) and len(file_list_before) != len(file_list_after):
    print("Uwaga: Liczba obrazów po augmentacji nie jest równa 5-krotności liczby obrazów przed augmentacją.")
    print("Sprawdź, czy wszystkie obrazy zostały poprawnie przetworzone.")
