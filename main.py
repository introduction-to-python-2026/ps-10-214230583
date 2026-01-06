import numpy as np
from PIL import Image
from scipy.signal import convolve2d
import matplotlib.pyplot as plt

# ספריות חדשות שביקשו בהוראות לניקוי רעשים
from skimage.filters import median
from skimage.morphology import ball

# --- הפונקציות שכתבנו קודם ---

def load_image(path):
    image = Image.open(path)
    return np.array(image)

def edge_detection(image_array):
    # 1. המרה לאפור
    gray_image = np.mean(image_array, axis=2)

    # 2. הגדרת פילטרים
    filter_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
    filter_y = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])

    # 3. קונבולוציה
    edge_x = convolve2d(gray_image, filter_x, mode='same', boundary='fill', fillvalue=0)
    edge_y = convolve2d(gray_image, filter_y, mode='same', boundary='fill', fillvalue=0)

    # 4. חישוב עוצמה (Magnitude)
    edge_mag = np.sqrt(edge_x**2 + edge_y**2)
    return edge_mag

# --- ביצוע ההוראות החדשות (שלב 3) ---

# 1. טעינת התמונה
filename = 'spongebob.png'
original_image = load_image(filename)

# 2. ניקוי רעשים (Denoising) - לפי הקוד שנתנו בתרגיל
print("Cleaning image noise...")
# הפונקציה median עם ball(3) מנקה רעשים תוך התחשבות בסביבה תלת-ממדית (RGB)
clean_image = median(original_image, ball(3))

# 3. הפעלת זיהוי הקצוות על התמונה הנקייה
print("Detecting edges...")
edge_mag = edge_detection(clean_image)

# 4. בחירת ערך סף (Threshold) והמרה לבינארי
# בהוראות ביקשו להסתכל על ההיסטוגרמה כדי לבחור ערך.
# נציג את ההיסטוגרמה כדי שתוכל לבחור מספר מתאים:
plt.figure()
plt.hist(edge_mag.flatten(), bins=50, log=True)
plt.title("Histogram of Edge Magnitude")
plt.show()

# --- כאן אתה צריך לבחור את המספר (הסף) ---
# ערכים נמוכים הם "רקע", ערכים גבוהים הם "קצוות".
# שנה את המספר 50 למספר שמתאים לתמונה שלך לפי הגרף
threshold_value = 50 

# יצירת תמונה בינארית (True/False או 0/1)
binary_edge_image = edge_mag > threshold_value

# 5. הצגה ושמירה
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.imshow(clean_image)
plt.title("Clean Image")

plt.subplot(1, 2, 2)
plt.imshow(binary_edge_image, cmap='gray')
plt.title(f"Binary Edges (Threshold={threshold_value})")
plt.show()

# שמירת התמונה כקובץ
# צריך להפוך את הבוליאני (True/False) למספרים (0-255) כדי לשמור
final_image_to_save = Image.fromarray((binary_edge_image * 255).astype(np.uint8))
final_image_to_save.save("edges_result.png")
print("Image saved as edges_result.png")
