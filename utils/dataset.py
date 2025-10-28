from torch.utils.data import Dataset
from PIL import Image
import os
import random
from torchvision import transforms

class UnpairedDataset(Dataset):
    def __init__(self, dir_old, dir_clean):
        self.old_images = [os.path.join(dir_old, x) for x in os.listdir(dir_old)]
        self.clean_images = [os.path.join(dir_clean, x) for x in os.listdir(dir_clean)]

        self.transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

    def __len__(self):
        return max(len(self.old_images), len(self.clean_images))

    def __getitem__(self, idx):
        old_path = random.choice(self.old_images)
        clean_path = random.choice(self.clean_images)

        old_img = self.transform(Image.open(old_path).convert("RGB"))
        clean_img = self.transform(Image.open(clean_path).convert("RGB"))

        return {'old': old_img, 'clean': clean_img}
