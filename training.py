##training##

import os
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

if torch.cuda.is_available():
    print(f'using gpu: {torch.cuda.get_device_name()}')
else:
    print('no gpu is available')



import glob
train_x = sorted(glob.glob("/kaggle/working/train/image/*"))
train_y = sorted(glob.glob("/kaggle/working/train/mask/*"))
valid_x = sorted(glob.glob("/kaggle/working/test/image/*"))
valid_y = sorted(glob.glob("/kaggle/working/test/mask/*"))


import os
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
train_loader = DataLoader(fundus_dataset(train_x, train_y), batch_size=2, shuffle=True, num_workers=2)         
test_loader = DataLoader(fundus_dataset(valid_x, valid_y), batch_size=2, shuffle=False, num_workers=2)


import torch
import numpy as np
import wandb
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Initialize W&B
wandb.init(project="lung", name="test run")

# Define optimizer and learning rate scheduler
lr = 1e-4
optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)  # Added weight decay
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=5, factor=0.1, verbose=True)

# Define your loss criterion
criterion = Dice_BCE()

# Set maximum epochs and early stopping patience
epochs = 1
early_stopping_patience = 10
best_test_loss = float('inf')
early_stopping_counter = 0

thresholds = [0.4, 0.5, 0.6]
# Training loop
for epoch in range(epochs):
    model.train()
    total_train_loss = 0.0
    
    for x_train, y_train in train_loader:
        x_train, y_train = x_train.to(device, dtype=torch.float32), y_train.to(device, dtype=torch.float32)
        
        # Forward pass
        outputs = model(x_train)
        loss = criterion(outputs, y_train)
        total_train_loss += loss.item()
        
        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
    avg_train_loss = total_train_loss / len(train_loader)

    # Test loop
    model.eval()
    total_test_loss = 0.0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for x_test, y_test in test_loader:
            x_test, y_test = x_test.to(device, dtype=torch.float32), y_test.to(device, dtype=torch.float32)
            test_outputs = model(x_test)
            test_loss = criterion(test_outputs, y_test)
            total_test_loss += test_loss.item()

            
            y_pred_prob = torch.sigmoid(test_outputs).cpu().numpy()
            y_test_flat = torch.sigmoid(y_test).cpu().numpy() 
            
            all_preds.append(y_pred_prob.flatten())
            all_labels.append(y_test_flat.flatten())

    avg_test_loss = total_test_loss / len(test_loader)
    all_preds_flat = np.concatenate(all_preds)
    all_labels_flat = np.concatenate(all_labels)

    # Metrics calculation
    threshold_metrics = {}
    for t in thresholds:
        binary_preds = (all_preds_flat > t).astype(np.int32)
        binary_labels = (all_labels_flat > t).astype(np.int32)
        
        accuracy = accuracy_score(binary_labels, binary_preds)
        precision = precision_score(binary_labels, binary_preds)
        recall = recall_score(binary_labels, binary_preds)
        dice_score = (2 * np.sum(binary_preds * binary_labels)) / (np.sum(binary_preds) + np.sum(binary_labels) + 1e-6)

    
        # Log threshold-specific metrics
        threshold_metrics[f"Accuracy (Threshold {t})"] = accuracy
        threshold_metrics[f"Precision (Threshold {t})"] = precision
        threshold_metrics[f"Recall (Threshold {t})"] = recall
        threshold_metrics[f"Dice Score (Threshold {t})"] = dice_score
    
   
    # Log metrics to W&B
    wandb.log({
        "Epoch": epoch + 1,
        "Training Loss": avg_train_loss,
        "Test Loss": avg_test_loss,
        "Learning Rate": scheduler.get_last_lr()[0],
        **threshold_metrics
    })

    print(f"Epoch [{epoch + 1}/{epochs}]")
    print(f"Training Loss: {avg_train_loss:.4f}, Test Loss: {avg_test_loss:.4f}")
    for t in thresholds:
        print(f"Threshold {t} -> Accuracy: {threshold_metrics[f'Accuracy (Threshold {t})']:.4f}, "
              f"Precision: {threshold_metrics[f'Precision (Threshold {t})']:.4f}, "
              f"Recall: {threshold_metrics[f'Recall (Threshold {t})']:.4f}, "
              f"Dice Score: {threshold_metrics[f'Dice Score (Threshold {t})']:.4f}")

    # Scheduler step
    scheduler.step(avg_test_loss)

    # Early stopping
    if avg_test_loss < best_test_loss:
        best_test_loss = avg_test_loss
        early_stopping_counter = 0  # Reset counter
        torch.save(model.state_dict(), "100ep_best_model.pth")  # Save the best model
    else:
        early_stopping_counter += 1
        if early_stopping_counter >= early_stopping_patience:
            print(f"Early stopping at epoch {epoch + 1}")
            break

# Finish the W&B run
wandb.finish()






