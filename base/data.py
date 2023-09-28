import os
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from torchvision import transforms as T

transform = T.Compose([
                    T.Resize(size=(384, 384)),
                    T.RandomHorizontalFlip(),
                    T.ToTensor(),
                    T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                    ])


class CustomDataset(Dataset):
    def __init__(self, csv_path, image_dir, transform=transform):
        self.data = pd.read_csv(csv_path, header=None)
        self.image_dir = image_dir
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_name = os.path.join(self.image_dir, self.data.iloc[idx, 0])
        image = Image.open(img_name)
        labels = torch.tensor(self.data.iloc[idx, 1:].values.astype(float), dtype=torch.float32)


        if self.transform:
            image = self.transform(image)

        return image, labels

# Paths and parameters
# csv_path = '/home/datdt/Desktop/Code/PAR/upar_challenge/data/phase1/train/train_2.csv'
# image_dir = '/home/datdt/Desktop/Code/PAR/upar_challenge/data'
# batch_size = 1

# # Create dataset and dataloader
# dataset = CustomDataset(csv_path, image_dir, transform=transform)
# dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=8)
# print(len(dataloader))
# # You can now use the dataloader for training
# for batch in dataloader:
#     images, labels = batch
#     # Your training code here
