import numpy as np
import torch
import os

from sklearn.metrics import confusion_matrix

from torch_dataset import TorchDataset # Assuming your TorchDataset class is in 'your_module.py'
from model import DeeperFCDNNFixed

# --- 1. Load the trained model ---
model_path = "best_model.pth"  # Replace with the actual path to your model file
model = DeeperFCDNNFixed(10)
try:
    state_dict = torch.load(model_path)
    model.load_state_dict(state_dict)
    model.eval() # Set to evaluation mode
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    print(f"Successfully loaded the trained model from: {model_path} and moved to {device}")
except FileNotFoundError:
    print(f"Error: Model file not found at {model_path}")
    exit()

# --- 3. Load the test dataset using your custom TorchDataset ---
test_data_dir = os.path.join("..", "..", "data", "input", "test")
try:
    test_dataset = TorchDataset(test_data_dir)
    print(f"Successfully created the test dataset from: {test_data_dir} with normalization.")
    label_map = test_dataset.get_label_map()
    class_names = list(label_map.keys())
    num_classes = len(class_names)
    print(f"Number of test samples: {len(test_dataset)}")
    print(f"Class names: {class_names}")
except FileNotFoundError:
    print(f"Error: Test data directory not found at {test_data_dir}")
    exit()

# --- 4. Create a DataLoader for your test dataset (for batching) ---
from torch.utils.data import DataLoader
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False) # No need to shuffle for testing
print("Successfully created the test DataLoader.")

# Now you have your model, normalization stats, test dataset, and test loader ready!

def evaluate_model(model, test_loader, device, num_classes, class_names):
    model.eval()  # Set the model to evaluation mode
    correct_predictions = 0
    total_samples = 0
    all_predicted = []
    all_labels = []

    # Disable gradient calculation during evaluation
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)

            # Add channel dimension if your model expects it (e.g., [batch_size, 1, height, width])
            if inputs.ndim == 3:
                inputs = inputs.unsqueeze(1)

            # Forward pass: get the model's output (logits)
            outputs = model(inputs)

            # Get the predicted class (the class with the highest probability)
            _, predicted = torch.max(outputs, 1)

            # Update counts of correct predictions and total samples
            total_samples += labels.size(0)
            correct_predictions += (predicted == labels).sum().item()

            # Store predictions and labels for calculating other metrics
            all_predicted.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    accuracy = correct_predictions / total_samples
    confusion = confusion_matrix(all_labels, all_predicted)

    precision = np.zeros(num_classes)
    for i in range(num_classes):
        tp = confusion[i, i]
        fp = np.sum(confusion[:, i]) - tp
        if (tp + fp) > 0:
            precision[i] = tp / (tp + fp)
        else:
            precision[i] = 0.0

    return accuracy, precision, confusion

# --- Run the evaluation ---
test_accuracy, test_precision, confusion_matrix_result = evaluate_model(model, test_loader, device, num_classes, class_names)

print(f'Test Accuracy: {test_accuracy * 100:.2f}%')
print('Test Precision per class:')
for i, class_name in enumerate(class_names):
    print(f'  {class_name}: {test_precision[i] * 100:.2f}%')
print('\nConfusion Matrix:')
print(confusion_matrix_result)