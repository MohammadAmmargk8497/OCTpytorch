import os
import torch
from tqdm.autonotebook import tqdm
from typing import Dict, List, Tuple
import wandb
import logger

def train_step(model: torch.nn.Module,
               dataloader: torch.utils.data.DataLoader,
               Accumulation_steps:int,
               loss_fn: torch.nn.Module,
               optimizer: torch.optim.Optimizer,
               device: torch.device,
               batch_cnt,
               exp_cnt,
               epoch)-> Tuple[float, float]: 
  
    model.train()

    # Setup train loss and train accuracy values
    train_loss, train_acc = 0, 0
 
    # Loop through data loader data batches
    for batch, (X, y) in tqdm(enumerate(dataloader), total =len(dataloader)):
        # Send data to target device
        X, y = X.to(device), y.to(device)

        # 1. Forward pass
        y_pred = model(X)

        # 2. Calculate  and accumulate loss
        loss = loss_fn(y_pred, y)
        exp_cnt +=  len(X)
        batch_cnt += 1
        train_loss += loss.item() 
        if ((batch_cnt + 1) % 25) == 0:
                logger.train_log(loss, exp_cnt, epoch)
        loss = loss/Accumulation_steps

        loss.backward()

        
        if ((batch + 1) % Accumulation_steps == 0) or (batch + 1 == len(dataloader)):
            
		# Update Optimizer
            optimizer.step()
        # 3. Optimizer zero grad
        
            optimizer.zero_grad()

        

        # 4. Loss backward
        
        # 5. Optimizer step
       

        # Calculate and accumulate accuracy metric across all batches
        y_pred_class = torch.argmax(torch.softmax(y_pred, dim=1), dim=1)
        train_acc += (y_pred_class == y).sum().item()/len(y_pred)

    # Adjust metrics to get average loss and accuracy per batch 
    train_loss = train_loss / len(dataloader)
    train_acc = train_acc / len(dataloader)
    return train_loss, train_acc

########Main func for Training#############

def train(model: torch.nn.Module, 
          train_dataloader: torch.utils.data.DataLoader,
          val_dataloader: torch.utils.data.DataLoader,
          Accumulation_steps,
          optimizer: torch.optim.Optimizer,
          loss_fn: torch.nn.Module,
          epochs: int,
          device: torch.device) -> Dict[str, List]:
 

  
  results = {"train_loss": [],
      "train_acc": [],
      "val_loss": [],
      "val_acc": []
      
  }
  batch_cnt=0
  exp_cnt=0  
  
  for epoch in range(epochs):
      wandb.watch(model, loss_fn, log="all", log_freq=10)
     
      train_loss, train_acc = train_step(model=model,
                                          dataloader=train_dataloader,
                                          Accumulation_steps=Accumulation_steps,
                                          loss_fn=loss_fn,
                                          optimizer=optimizer,
                                          device=device,
                                          batch_cnt=batch_cnt,
                                          exp_cnt=exp_cnt,
                                          epoch=epoch)
      
      model.eval()
      for batch, (X, y) in enumerate(val_dataloader):
          
          X, y = X.to(device), y.to(device)

        # 1. Forward pass
          y_pred = model(X)

          loss = loss_fn(y_pred, y)

          val_loss += loss.item()

          y_pred_class = torch.argmax(torch.softmax(y_pred, dim=1), dim=1)
          val_acc += (y_pred_class == y).sum().item()/len(y_pred)

    # Adjust metrics to get average loss and accuracy per batch 
          val_loss = train_loss / len(val_dataloader)
          val_acc = train_acc / len(val_dataloader)

          


          
      
      
      

      # Print out what's happening
      print(
          f"Epoch: {epoch+1} | "
          f"train_loss: {train_loss:.4f} | "
          f"train_acc: {train_acc:.4f} | "
          f"val_loss: {val_loss:.4f} | "
          f"val_acc: {val_acc:.4f} | "
          )

      # Update results dictionary
      results["train_loss"].append(train_loss)
      results["train_acc"].append(train_acc)
      results["train_loss"].append(val_loss)
      results["train_acc"].append(val_acc)

     
  
  return results
                                                           
  