import os
from torchvision import datasets, transforms
from torch.utils.data import DataLoader



NUM_WORKERS= os.cpu_count()

def create_dataloaders(train_dir :str, test_dir : str, val_dir : str, transform: transforms.Compose, batch_size:int, num_workers:int= NUM_WORKERS ):
    
  train_data = datasets.ImageFolder(train_dir, transform = transform)
  test_data = datasets.ImageFolder(test_dir, transform = transform)
  val_data = datasets.ImageFolder(val_dir, transform = transform)
  class_names = train_data.classes

  train_dataloader = DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory = True)
  test_dataloader = DataLoader(test_data, batch_size = batch_size, shuffle=True, num_workers=num_workers, pin_memory = True)
  val_dataloader = DataLoader(val_data, batch_size = batch_size, shuffle=True, num_workers=num_workers, pin_memory = True)

  return train_dataloader, test_dataloader, val_dataloader, class_names



###########################to delete
# train_dir = "C:/Users/Umair/Desktop/Balanced/train"
# test_dir = "C:/Users/Umair/Desktop/Balanced/test"
# val_dir  = "C:/Users/Umair/Desktop/Balanced/val"
# data_transform = transforms.Compose([
#   transforms.Resize((256, 256)),
#   transforms.ToTensor()
# ])

#   # Create DataLoaders with help from data_setup.py
# train_dataloader, test_dataloader, val_dataloader, class_names = create_dataloaders(
#     train_dir=train_dir,
#     test_dir=test_dir,
#     val_dir=val_dir,
#     transform=data_transform,
#     batch_size=32
# )
