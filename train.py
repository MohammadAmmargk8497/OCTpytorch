import os
import torch
import dataloader, engiine, model_builder, utils
import icecream as ic

# from torchvision import transforms
import torchvision.transforms as transforms

def train(config):
  
  print(type(config))
  #ic(config)
  # Setup hyperparameters
  # NUM_EPOCHS, BATCH_SIZE, LEARNING_RATE, Accumulation_steps = config.config()

  # Setup directories
  train_dir = "C:/Users/Umair/Desktop/Balanced/train"
  test_dir = "C:/Users/Umair/Desktop/Balanced/test"
  val_dir  = "C:/Users/Umair/Desktop/Balanced/val"
  # Setup target device
  device = "cuda" if torch.cuda.is_available() else "cpu"

  # Create transforms
  data_transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor()
  ])

  # Create DataLoaders with help from data_setup.py
  train_dataloader, test_dataloader, val_dataloader, class_names = dataloader.create_dataloaders(
      train_dir=train_dir,
      test_dir=test_dir,
      val_dir=val_dir,
      transform=data_transform,
      batch_size=config.batch_size
  )

  # Create model with help from model_builder.py
  model = model_builder.mobilevit_xxs(
      num_classes=len(class_names)
  ).to(device)

  # Set loss and optimizer
  loss_fn = torch.nn.CrossEntropyLoss()
  optimizer = torch.optim.Adam(model.parameters(),
                               lr=config.learning_rate)
  scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.1, patience=2, verbose=True)

  # Start training with help from engine.py
  engiine.train(model=model,
               train_dataloader=train_dataloader,
               Accumulation_steps = config.Accumulation_steps,
               loss_fn=loss_fn,
               optimizer=optimizer,
               epochs=config.epochs,
               device=device)

  # Save the model with help from utils.py
  utils.save_model(model=model,
                   target_dir="models",
                   model_name="05_going_modular_script_mode_tinyvgg_model.pth")