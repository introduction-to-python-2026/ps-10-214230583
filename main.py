# main.py
import numpy as np
import matplotlib.pyplot as plt
from skimage.filters import median
from skimage.morphology import ball
from PIL import Image

# ייבוא הפונקציות שלנו מהקובץ השני
from image_utils import load_image, edge_detection

# 1. טעינת התמונה
filename = 'spongebob.png'
original_image = load_image(filename)

# 2. ניקוי רעשים
print("Cleaning image noise...")
clean_image = median(original_image, ball(3))

# 3. זיהוי קצוות
print("Detecting edges...")
edge_mag = edge_detection(clean_image)

# 4. המרה לבינארי (לפי הסף שבחרתם - 50)
threshold = 50
binary_edge_image = edge_mag > threshold

# 5. הצגה (אופציונלי - אם רוצים לראות שוב)
plt.imshow(binary_edge_image, cmap='gray')
plt.title(f"Final Edges (Threshold={threshold})")
plt.show()

# 6. שמירת התמונה
final_image = Image.fromarray((binary_edge_image * 255).astype(np.uint8))
final_image.save("edges_result.png")
print("Image saved successfully as 'edges_result.png'")
