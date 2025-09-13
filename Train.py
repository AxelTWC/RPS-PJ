# --- 
# Remarks :
# Speaking for the future , this small lightweight project is intended to be a learning process for me , so every single line of code should have a comment explaining what it does
# I will try to keep the code as clean and organized as possible , so it will be easier for me to understand it in the future
# This is the real beginning of my touch on AI and Deep Learning , along side my studies.
# ---

# ---------------------------
# IMPORTS
# ---------------------------


import torch # Core PyTorch library (tensors, GPU ops, training utilities)
from torch import nn, optim  # 'nn' = neural network layers, 'optim' = optimization algorithms
from torch.utils.data import DataLoader, random_split  
# DataLoader = batches data efficiently
# random_split = easily split dataset into train/validation subsets

from torchvision import datasets, transforms, models
# datasets = ready-to-use image datasets / loaders
# transforms = preprocessing + augmentation for images
# models = pretrained CNNs like ResNet, VGG, etc.

import os 

# ---------------------------
# CONFIGURATION
# ---------------------------
data_dir = "./rpsimages"       # Folder where Rock–Paper–Scissors dataset is stored (with subfolders rock/paper/scissors)
batch_size = 32          # Number of images processed together in one forward/backward pass
num_epochs = 10          # How many times we loop through the whole training dataset
lr = 1e-3                # Learning rate (step size for optimizer) 
num_classes = 3          # Rock, Paper, Scissors |  3 categories
use_transfer = True      # If True, we use transfer learning with ResNet18. If False, we build a CNN from scratch. Will Cover This In Another Project

# Select device (GPU if available, otherwise CPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# ---------------------------
# DATA TRANSFORMS
# ---------------------------

# Adds variations to our data as real life data are skewed and not perfect anyways:

train_transform = transforms.Compose([
    transforms.Resize((128, 128)),         # Resize all images to 128x128 (standard input size for CNN)
    transforms.RandomHorizontalFlip(),     # Randomly flip images horizontally (augmentation)
    transforms.RandomRotation(20),         # Randomly rotate image up to ±20 degrees (augmentation)
    transforms.ColorJitter(                # Randomly adjust brightness/contrast/saturation (augmentation)
        brightness=0.2, contrast=0.2, saturation=0.2
    ),
    transforms.ToTensor(),                 # Convert PIL (Python imaging library) image → PyTorch tensor (values scaled to [0,1])
])


# Validation transforms (no augmentation, just resize + tensor conversion)
val_transform = transforms.Compose([
    transforms.Resize((128, 128)),         # Same resize
    transforms.ToTensor(),                 # Convert to tensor | Remarks: multidimensional array (like NumPy arrays, but with extra features) , PyTorch tensors are the core data structure used in deep learning. Models only understand tensors, not Python lists or PIL images.

# 0D tensor → single number

# 1D tensor → vector

# 2D tensor → matrix

# 3D tensor → color image (channels × height × width) --- Channel = RGB , so 3 channels

# 4D tensor → batch of images (batch × channels × height × width) -- Assume the same video explaining computation of CPU vs GPU on a painting. A batch is basically showing that GPU can do multiple machine shooting paint at once.

])

# ---------------------------
# DATASET & LOADERS
# ---------------------------

# Load dataset from folders.
# Assumes folder structure: rps/rock, rps/paper, rps/scissors
dataset = datasets.ImageFolder(root=data_dir, transform=train_transform)

# Split dataset into training and validation sets (80% train, 20% validation) <-- Learnt from Stanford Online Course saying we need training set and validation set.
train_size = int(0.8 * len(dataset))       # 80% of total dataset
val_size = len(dataset) - train_size       # remaining 20%
train_ds, val_ds = random_split(dataset, [train_size, val_size])

# Change transform of validation dataset → val_transform (no augmentation, just resize+tensor)
val_ds.dataset.transform = val_transform

# Create DataLoader for batching and shuffling
train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
# shuffle=True → randomize order of data each epoch

val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
# shuffle=False for validation → keep it consistent


# ---------------------------
# MODEL SETUP
# ---------------------------

if use_transfer:
    # Load a pretrained ResNet18 model (trained on ImageNet, ~1M images)
    model = models.resnet18(pretrained=True)

    # Freeze all pretrained layers → we won’t update their weights during training | A weight is like a knob that controls strongly one neuron influence the next neuron
    for param in model.parameters():
        param.requires_grad = False

    # Replace the last fully connected layer with a new one for 3 classes
    # (original ResNet has 1000 output classes for ImageNet)
    model.fc = nn.Linear(model.fc.in_features, num_classes)


else: #Going to Expand this Knowledge on Next Project
    # Define a custom CNN from scratch (simpler, fewer layers)
    model = nn.Sequential(
        nn.Conv2d(3, 32, 3, padding=1),  # Conv layer: 3 (RGB) → 32 feature maps
        nn.ReLU(),                       # Non-linear activation
        nn.MaxPool2d(2),                 # Downsample by factor of 2
        
        nn.Conv2d(32, 64, 3, padding=1), # Conv: 32 → 64 feature maps
        nn.ReLU(),                       # Non-linearity
        nn.MaxPool2d(2),                 # Downsample
        
        nn.Flatten(),                    # Flatten to 1D vector for fully connected layers
        nn.Linear(64*32*32, 128), nn.ReLU(),  # Dense hidden layer (64×32×32 inputs → 128 outputs)
        nn.Dropout(0.3),                 # Dropout (randomly zeroes 30% of activations for regularization)
        nn.Linear(128, num_classes)      # Final layer → 3 output classes
    )

# Move model to GPU (if available) or CPU
model = model.to(device)

# ---------------------------
# TRAINING UTILITIES
# ---------------------------

criterion = nn.CrossEntropyLoss()
# CrossEntropyLoss = classification loss function
# (combines softmax + negative log likelihood)

optimizer = optim.Adam(model.parameters(), lr=lr)
# Adam optimizer (adaptive learning rate) with learning rate = lr
# Adam = Adaptive Moment Estimation --- kinda like a smart thing that tunes all weight automatically.
# Learning rate = how big a step we take when updating weights. Too high → unstable training. Too low → slow training.


# ---------------------------
# TRAINING FUNCTION (1 epoch)
# ---------------------------
def train_one_epoch(): # Epoch in this case is basically iteration 1: Forward + Backward + Update for image 1-32 , and that is Batch 1 , Assume there are 320 images , so 10 batches per epoch.

    model.train() # Set model in training mode (enables dropout, batchnorm updates)
    total, correct, running_loss = 0, 0, 0

    # Loop over training batches
    for imgs, labels in train_loader:
        imgs, labels = imgs.to(device), labels.to(device)  # Move data to GPU/CPU

        optimizer.zero_grad()            # Reset gradients before backward pass
        outputs = model(imgs)            # Forward pass → predictions
        loss = criterion(outputs, labels)# Compute loss between predictions & ground truth
        loss.backward()                  # Backward pass → compute gradients
        optimizer.step()                 # Update weights
        
        running_loss += loss.item() * imgs.size(0)  # Accumulate total loss
        _, preds = torch.max(outputs, 1)            # Get predicted class index
        correct += (preds == labels).sum().item()   # Count correct predictions
        total += labels.size(0)                     # Count total samples
    
    return running_loss/total, correct/total        # Average loss, accuracy


# | Step                 | Analogy                                                              |
# | -------------------- | -------------------------------------------------------------------- |
# | Images + labels      | Showing pictures and saying “this is rock”                           |
# | Model prediction     | Kid guesses “maybe this is paper?”                                   |
# | Loss                 | Teacher says “wrong, you were off by this much”                      |
# | Backprop + optimizer | Kid adjusts understanding of features (e.g., shape of fist, fingers) |
# | Epochs               | Repeating this with many images until the kid learns reliably        |


# ---------------------------
# VALIDATION FUNCTION
# ---------------------------

def validate():
    model.eval()  # Set model to evaluation mode (disables dropout, batchnorm updates)
    total, correct, running_loss = 0, 0, 0
    
    with torch.no_grad():               # Disable gradient tracking (saves memory + speed)
        for imgs, labels in val_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            outputs = model(imgs)                     # Forward pass
            loss = criterion(outputs, labels)         # Compute loss
            running_loss += loss.item() * imgs.size(0)
            _, preds = torch.max(outputs, 1)          # Predicted class
            correct += (preds == labels).sum().item()
            total += labels.size(0)
    
    return running_loss/total, correct/total          # Return average loss, accuracy

# ---------------------------
# TRAINING LOOP (multiple epochs)
# ---------------------------

best_val_acc = 0    # Track best validation accuracy so far
for epoch in range(1, num_epochs+1):    # Loop through epochs (1 → num_epochs)
    train_loss, train_acc = train_one_epoch()   # Train for one epoch
    val_loss, val_acc = validate()              # Evaluate on validation set
    
    # Print stats for this epoch
    print(f"Epoch {epoch}/{num_epochs} "
          f"Train Loss: {train_loss:.4f} Acc: {train_acc*100:.2f}% "
          f"Val Loss: {val_loss:.4f} Acc: {val_acc*100:.2f}%")
    
    # Save model if validation accuracy improved
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        torch.save(model.state_dict(), "rps_best.pt")   # Save only model weights
        print("Saved new best model!")

# After training finishes
print("Training complete. Best Val Acc: {:.2f}%".format(best_val_acc*100))