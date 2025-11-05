import json
import matplotlib.pyplot as plt
import numpy as np

# Initialize lists to store data
train_loss, val_loss = [], []
train_acc, val_acc = [], []
train_top5_acc, val_top5_acc = [], []  # To store Top-5 accuracy
epochs_train, epochs_val = [], []
lr_train, lr_val = [], []  # To store learning rates

# Read the JSON lines
with open('D:/TFG/entrenamiento_stcgn/stgcn++/stgcn++_ntu120_xsub_3dkp/j/20250526_164625.log.json', 'r') as f:
    for line in f:
        try:
            # Parse the JSON line
            entry = json.loads(line.strip())
            
            # Handle the train data
            if entry['mode'] == 'train':
                train_loss.append(entry['loss'])
                train_acc.append(entry['top1_acc'])
                train_top5_acc.append(entry['top5_acc'])  # Collect Top-5 Accuracy for training
                epochs_train.append(entry['epoch'])
                lr_train.append(entry['lr'])
            
            # Handle the validation data
            elif entry['mode'] == 'val':
                val_loss.append(entry.get('loss', None))  # Val may not always have a 'loss' value
                val_acc.append(entry['top1_acc'])
                val_top5_acc.append(entry['top5_acc'])  # Collect Top-5 Accuracy for validation
                epochs_val.append(entry['epoch'])
                lr_val.append(entry['lr'])  # Store learning rate for validation
            
        except KeyError as e:
            print(f"Missing key: {e} in entry: {line.strip()}")

# Filter out None values from val_loss to avoid errors in processing
val_loss_filtered = [loss for loss in val_loss if loss is not None]

# Smooth the loss curves (only if there are enough points to smooth)
train_loss_smooth = np.convolve(train_loss, np.ones(5)/5, mode='valid')
if val_loss_filtered:
    val_loss_smooth = np.convolve(val_loss_filtered, np.ones(5)/5, mode='valid')
else:
    val_loss_smooth = []

# Plotting
plt.figure(figsize=(12, 6))

# Plot Training and Validation Loss
plt.subplot(1, 4, 1)
plt.plot(epochs_train[:len(train_loss_smooth)], train_loss_smooth, label='Train Loss', color='blue', linewidth=2)
if val_loss_smooth:  # Only plot validation loss if it is available
    plt.plot(epochs_val[:len(val_loss_smooth)], val_loss_smooth, label='Val Loss', color='orange', linewidth=2)
plt.xlabel('Epoch', fontsize=16)
plt.ylabel('Loss', fontsize=16)
plt.title('Training Loss', fontsize=18)
plt.legend(fontsize=14)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.grid(True)

# Plot Training and Validation Top-1 Accuracy
plt.subplot(1, 4, 2)
plt.plot(epochs_train, train_acc, label='Train Accuracy', color='green')
plt.plot(epochs_val, val_acc, label='Val Accuracy', color='red')
plt.xlabel('Epoch')
plt.ylabel('Top-1 Accuracy')
plt.legend()
plt.title('Training and Validation Top-1 Accuracy')
plt.grid(True)

# Plot Training and Validation Top-5 Accuracy
plt.subplot(1, 4, 3)
plt.plot(epochs_train, train_top5_acc, label='Train Top-5 Accuracy', color='purple')
plt.plot(epochs_val, val_top5_acc, label='Val Top-5 Accuracy', color='brown')
plt.xlabel('Epoch')
plt.ylabel('Top-5 Accuracy')
plt.legend()
plt.title('Training and Validation Top-5 Accuracy')
plt.grid(True)

# Plot Learning Rate
plt.subplot(1, 4, 4)
plt.plot(epochs_train, lr_train, label='Train LR', color='cyan', linewidth=2)
plt.plot(epochs_val, lr_val, label='Val LR', color='magenta', linewidth=2)
plt.xlabel('Epoch', fontsize=16)
plt.ylabel('Learning Rate', fontsize=16)
plt.title('Learning Rate', fontsize=18)
plt.legend(fontsize=14)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.grid(True)

# Improve layout and show the plots
plt.tight_layout()
plt.show()
