import cv2
import numpy as np
import mahotas
from skimage.morphology import skeletonize


def compute_fourier_descriptor(skeleton):
    points = np.column_stack(np.where(skeleton == 255))
    complex_points = points[:, 1] + 1j * points[:, 0]

    fourier_descriptor = np.fft.fft(complex_points)
    fourier_descriptor = np.abs(fourier_descriptor)
    descriptor = fourier_descriptor[1:11]

    return descriptor / (np.linalg.norm(descriptor) + 1e-8)


def augment_data(person, target_samples=10, noise_factor=0.05):
    augmented_data = person.copy()

    np.random.seed(0)

    while len(augmented_data) < target_samples:
        # Wybierz losowo dwie próbki do interpolacji
        idx1, idx2 = np.random.choice(len(person), 2, replace=False)
        sample1, sample2 = np.array(augmented_data[idx1]), np.array(
            augmented_data[idx2]
        )
        # Interpolacja liniowa
        alpha = np.random.rand()
        new_sample = alpha * sample1 + (1 - alpha) * sample2
        # Dodanie szumu
        noise = (
            np.random.randn(*new_sample.shape)
            * noise_factor
            * np.std(augmented_data, axis=0)
        )
        new_sample = new_sample + noise
        # Konwersja na listę i dodanie do danych
        augmented_data.append(new_sample.tolist())

    return augmented_data[:target_samples]


def filter_skeleton(skeletonm, size):
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(skeleton)

    filtered_skeleton = np.zeros_like(skeleton)
    min_size = 20  # minimalna liczba pikseli w szkielecie
    for i in range(1, num_labels):  # pomiń tło (label 0)
        if stats[i, cv2.CC_STAT_AREA] >= min_size:
            filtered_skeleton[labels == i] = 255

    return filtered_skeleton


def delete_background(image, eps):
    for idx_row, row in enumerate(image):
        for idx_col, pixel in enumerate(row):
            b, g, r = pixel
            avg = (int(b) + int(g) + int(r)) // 3
            if (
                b in range(avg - eps, avg + eps)
                and g in range(avg - eps, avg + eps)
                and r in range(avg - eps, avg + eps)
            ):
                image[idx_row][idx_col] = [0, 0, 0]
    return image


scale_factor = 0.25
gray = cv2.resize(
    cv2.cvtColor(image, cv2.COLOR_BGR2GRAY),
    (0, 0),
    fx=scale_factor,
    fy=scale_factor,
    interpolation=cv2.INTER_AREA,
)


gray = cv2.medianBlur(gray, 5)
denoised = cv2.bilateralFilter(image, d=9, sigmaColor=75, sigmaSpace=75)

gray = cv2.bilateralFilter(gray, d=11, sigmaColor=100, sigmaSpace=25)
clahe = cv2.createCLAHE(clipLimit=3, tileGridSize=(8, 8))
gray = clahe.apply(gray)

#####################
# Image processing. #
# Binarization. Adaptive method #
#################################
binary = cv2.adaptiveThreshold(
    gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 15, 5
)
binary = 255 - cv2.medianBlur(binary, 5)

skeleton = skeletonize(255 - binary)
skeleton = skeleton.astype(np.uint8) * 255
skeleton = filter_skeleton(skeleton, 20)

y, x = np.where(skeleton == 255)
points = np.column_stack((x, y)).astype(np.int32)
hull = cv2.convexHull(points)

cv2.imwrite("Noise15.png", gray1)
cv2.imshow("Binary", gray1)
cv2.waitKey(0)
cv2.destroyAllWindows()

exit()


print("Fourier Descriptors:", compute_fourier_descriptor(skeleton))

cv2.imshow("Binary", skeleton)
cv2.waitKey(0)
cv2.destroyAllWindows()

exit()

skeleton = skeletonize(255 - binary)
skeleton = skeleton.astype(np.uint8) * 255
output_image = 255 - skeleton

# === Hu and Zernike moments (dla obrazu binarnego) ===
hu_moments = cv2.HuMoments(cv2.moments(binary)).flatten()

radius = min(binary.shape) // 2
center = (binary.shape[1] // 2, binary.shape[0] // 2)
zernike_patch = binary[
    center[1] - radius : center[1] + radius, center[0] - radius : center[0] + radius
]
zernike_moments = mahotas.features.zernike_moments(zernike_patch, radius)

print("Hu Moments:", (hu_moments.round(5)))
print("Zernike Moments:", (zernike_moments.round(5)))

"""

"""
image = cv2.resize(
    image, (0, 0), fx=scale_factor, fy=scale_factor, interpolation=cv2.INTER_AREA
)
hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
lower = np.array([0, 0, 180])
upper = np.array([255, 40, 255])
mask = cv2.inRange(hsv, lower, upper)
mask_inv = cv2.bitwise_not(mask)
result = cv2.bitwise_and(image, image, mask=mask_inv)
"""

"""
y, x = np.where(binary == 0)
points = np.column_stack((x, y)).astype(np.int32)
hull = cv2.convexHull(points)

output = binary.copy()
cv2.drawContours(output, [hull], -1, (0, 255, 0), 2)
cv2.imshow("Convex Hull", output)
cv2.waitKey(0)
cv2.destroyAllWindows()

exit()

# Skeletonization.
skeleton = skeletonize(255 - binary)
skeleton = skeleton.astype(np.uint8) * 255
output_image = 255 - skeleton
convex_hull = skeleton

cv2.imshow("Convex Hull", skeleton)
cv2.waitKey(0)
cv2.destroyAllWindows()

exit()

y, x = np.where(convex_hull == 255)
points = np.column_stack((x, y)).astype(np.int32)
hull = cv2.convexHull(points)


# Cechy pomocnicze.
# Pole powierzchni otoczki wypukłej.
hull_area = cv2.contourArea(hull)
# Długość konturu otoczki wypukłej.
perimeter = cv2.arcLength(hull, True)
# Współrzędne prostokąta otaczającego.
x, y, w, h = cv2.boundingRect(hull)
# Pole powierzchni prostokąta otaczającego.
rect_area = w * h
