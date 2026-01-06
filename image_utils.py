import numpy as np
from PIL import Image
from scipy.signal import convolve2d  # <--- השורה החדשה שהוספנו

def load_image(path):
    image = Image.open(path)
    return np.array(image)

filename = 'spongebob.png'
image_array = load_image(filename)

print("הספרייה נטענה בהצלחה והתמונה מוכנה לעיבוד")


def detection_edge(image_array):
    # 1. המרה לתמונה אפורה (Grayscale)
    # חישוב הממוצע של 3 ערוצי הצבע (R, G, B) עבור כל פיקסל
    # axis=2 אומר שאנחנו ממצעים את העומק של המערך
    gray_image = np.mean(image_array, axis=2)

    # 2. בניית הפילטרים (Kernels) למציאת שינויים
    # פילטר לשינויים בכיוון האופקי (Horizontal) - מוצא קווים אנכיים
    # (Sobel Filter X דוגמה לקרויה)
    filter_x = np.array([[-1, 0, 1],
                         [-2, 0, 2],
                         [-1, 0, 1]])

    # פילטר לשינויים בכיוון האנכי (Vertical) - מוצא קווים אופקיים
    # (Sobel Filter Y דוגמה לקרויה)
    filter_y = np.array([[-1, -2, -1],
                         [0,  0,  0],
                         [1,  2,  1]])

    # 3. הפעלת הפילטרים באמצעות convolve2d
    # mode='same' מבטיח שהתמונה החדשה תישאר באותו גודל כמו המקורית
    # boundary='fill', fillvalue=0 מתייחס להוראת ה"padding 0" (מילוי אפסים בקצוות)
    edge_x = convolve2d(gray_image, filter_x, mode='same', boundary='fill', fillvalue=0)
    edge_y = convolve2d(gray_image, filter_y, mode='same', boundary='fill', fillvalue=0)

    # 4. חישוב המגניטודה (Magnitude) לפי הנוסחה
    edge_mag = np.sqrt(edge_x**2 + edge_y**2)

    return edge_mag

# --- חלק לבדיקה (אופציונלי) ---
# נניח שכבר טענת את התמונה מהשלב הקודם למשתנה spongebob_array
# result_image = detection_edge(spongebob_array)
# print(f"Output shape: {result_image.shape}")
