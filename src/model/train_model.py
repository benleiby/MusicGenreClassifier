import torch
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
from torchvision import datasets
import model
import torch_dataset
from torch import nn, optim
import os

train_dataset =  torch_dataset.TorchDataset(os.path.join("..", "..", "data", "input", "train"))
val_dataset = torch_dataset.TorchDataset(os.path.join("..", "..", "data", "input", "val"))

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

# --- Hyperparameters ---
num_epochs = 100
learning_rate = 0.00005
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- Initialization ---
model = model.DeeperFCDNNFixed(num_classes=10).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=0.0001)
scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.2, patience=5)

# --- Training Loop ---
best_val_accuracy = 0.0

for epoch in range(num_epochs):
    model.train()
    train_loss = 0.0
    train_correct = 0
    total_train = 0

    for inputs, labels in train_loader:
        inputs = inputs.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        train_loss += loss.item() * inputs.size(0)
        _, predicted = torch.max(outputs.data, 1)
        total_train += labels.size(0)
        train_correct += (predicted == labels).sum().item()

    train_loss = train_loss / total_train
    train_accuracy = 100 * train_correct / total_train

    # --- Validation Loop ---
    model.eval()
    val_loss = 0.0
    val_correct = 0
    total_val = 0

    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            loss = criterion(outputs, labels)

            val_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs.data, 1)
            total_val += labels.size(0)
            val_correct += (predicted == labels).sum().item()

    val_loss = val_loss / total_val
    val_accuracy = 100 * val_correct / total_val

    scheduler.step(val_accuracy)  # Step the scheduler with the validation accuracy

    print(f'Epoch [{epoch+1}/{num_epochs}], '
          f'Train Loss: {train_loss:.4f}, Train Acc: {train_accuracy:.2f}%, '
          f'Val Loss: {val_loss:.4f}, Val Acc: {val_accuracy:.2f}%')

    # Save the best model based on validation accuracy
    if val_accuracy > best_val_accuracy:
        best_val_accuracy = val_accuracy
        torch.save(model.state_dict(), 'best_model.pth')
        print(f'Best model saved with Val Acc: {best_val_accuracy:.2f}%')

print('Finished Training')
print(f'Best Validation Accuracy: {best_val_accuracy:.2f}%')