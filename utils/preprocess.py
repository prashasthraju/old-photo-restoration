import PIL
from PIL import Image
import os

print(PIL.__version__)

def preprocess_images(input_dir, output_dir, size=(256, 256)):
    os.makedirs(output_dir, exist_ok=True)
    for img_name in os.listdir(input_dir):
        try:
            img_path = os.path.join(input_dir, img_name)
            img = Image.open(img_path).convert("RGB")
            img = img.resize(size)
            img.save(os.path.join(output_dir, img_name))
        except:
            continue

if __name__ == "__main__":
    preprocess_images("data/old", "data/old_preprocessed")
    preprocess_images("data/clean", "data/clean_preprocessed")
