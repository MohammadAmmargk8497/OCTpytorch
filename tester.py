import os
import torch
from tqdm.auto import tqdm
from typing import Dict, List, Tuple

def test_step(model: torch.nn.Module, 
              test_dataloader: torch.utils.data.DataLoader, 
              loss_fn: torch.nn.Module,
              device: torch.device) -> Tuple[float, float]:
  

  
  model.eval() 

  
  test_loss, test_acc = 0, 0

  
  with torch.inference_mode():
      
      for X, y in enumerate(test_dataloader):
          
          X, y = X.to(device), y.to(device)

          
          test_pred_logits = model(X)

          
          loss = loss_fn(test_pred_logits, y)
          test_loss += loss.item()

          
          test_pred_labels = test_pred_logits.argmax(dim=1)
          test_acc += ((test_pred_labels == y).sum().item()/len(test_pred_labels))

  
  test_loss = test_loss / len(test_dataloader)
  test_acc = test_acc / len(test_dataloader)
  return test_loss, test_acc





# def test(model: torch.nn.Module, 
#          test_dataloader: torch.utils.data.DataLoader, 
#          optimizer: torch.optim.Optimizer,
#          loss_fn: torch.nn.Module,
#          epochs: int,
#          device: torch.device) -> Dict[str, List]:
 

  
#   results = {"train_loss": [],
#       "train_acc": [],
#       "test_loss": [],
#       "test_acc": []
#   }

  
#   for epoch in tqdm(range(epochs)):
#       test_loss, test_acc = test_step(model=model,
#                                           dataloader=test_dataloader,
#                                           loss_fn=loss_fn,
#                                           optimizer=optimizer,
#                                           device=device)
      

#       # Print out what's happening
#       print(
#           f"Epoch: {epoch+1} | "
#           f"train_loss: {test_loss:.4f} | "
#           f"train_acc: {test_acc:.4f} | "
#           )

#       # Update results dictionary
#       results["train_loss"].append(test_loss)
#       results["train_acc"].append(test_acc)
     
  
#   return results