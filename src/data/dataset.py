import cv2
import torch

from pathlib import Path
from torch.utils.data import Dataset

class ChestXrayDataSet(Dataset):
    """ 
        PyTorch Dataset for chest X-ray images.

        Each samples returns : 
        - Image tensor of shape (1, H, W)
        - Binary label : 
            0 = normal lung
            1 = tuberculosis
    """

    def __init__(self, root_dir, transform = None):
        self.root_dir = Path(root_dir)
        self.transform = transform

        self.samples = []
        self.labels = []

        # Assign labels based on folder names
        for label, class_name in enumerate(["normal", "tuberculosis"]):
            class_dir = self.root_dir / class_name
            for image_path in class_dir.glob("*.png"):
                self.samples.append(image_path)
                self.labels.append(label)
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        """
        Return a single sample for training

        Notes:
        - Images are converted to PyTorch tensors because CNNs operate on tensors, not raw images files.
            Shape is (1, H, W) where: 
                1 = number of channels ( grayscale)
                H = image height
                W = width
        """

        image_path = self.samples[idx]
        label = self.labels[idx]

        image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
        image = image.astype("float32")/255 # Normalize

        # Convert to tensor and add channel dimension
        # (H,W) -> ( 1, H, W) for CNN
        image = torch.from_numpy(image).unsqueeze(0)

        if self.transform:
            image = self.transform(image)

        return image,torch.tensor(label, dtype = torch.float32)