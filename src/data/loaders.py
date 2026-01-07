from torch.utils.data import DataLoader, random_split
from src.data.dataset import ChestXrayDataSet

def build_dataloaders(data_dir, batch_size = 15, val_split = 0.2):
    """
        Builds PyTorch DataLoader for training and validation
        Parameters: 
            - data_dir => path to processed dataset directory.
            - batch_size => number of images per batch
            - val_split => fraction of data reserved for validation
    """
    
    dataset = ChestXrayDataSet(data_dir)

    val_size = int(len(dataset) * val_split)
    train_size = len(dataset) - val_size

    train_ds, val_ds = random_split(dataset,[train_size,val_size])

    train_loader = DataLoader(
        train_ds,
        batch_size = batch_size,
        shuffle = True, # Improve generalization
        num_workers = 0
    )

    val_loader = DataLoader(
        val_ds,
        batch_size = batch_size,
        shuffle = False,
        num_workers = 0
    )

    return train_loader,val_loader