import numpy as np
from PIL import Image
from scipy.signal import convolve2d  # <--- השורה החדשה שהוספנו

def load_image(path):
    image = Image.open(path)
    return np.array(image)

filename = 'spongebob.png'
image_array = load_image(filename)

print("הספרייה נטענה בהצלחה והתמונה מוכנה לעיבוד")

def edge_detection(image):
    pass # Replace the `pass` with your code
