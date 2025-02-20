import torch
from sklearn.metrics import confusion_matrix, classification_report
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from model import LSTMModel, GestureDataset, calculate_model_size
from data_loader import GestureDataLoader

def evaluate_model(model, test_loader, device='cuda'):
    """Evaluate model on test set"""
    model.eval()
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for features, labels in test_loader:
            features, labels = features.to(device), labels.to(device)
            outputs = model(features)
            _, predicted = torch.max(outputs.data, 1)
            
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    return np.array(all_preds), np.array(all_labels)

def plot_confusion_matrix(y_true, y_pred, classes):
    """Plot confusion matrix"""
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=classes, yticklabels=classes)
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.show()

def main():
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load data from correct directories
    data_loader = GestureDataLoader(
        raw_data_dir='data/raw',
        processed_data_dir='data/processed',
        windowed_data_dir='data/windowed'
    )
    data = data_loader.prepare_data()
    
    # Create test dataset and loader
    test_dataset = GestureDataset(*data['test'])
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=32)
    
    # Load model
    model = LSTMModel().to(device)
    model.load_state_dict(torch.load('models/lstm_model.pth'))
    
    # Evaluate
    predictions, labels = evaluate_model(model, test_loader, device)
    
    # Get class names
    classes = list(data['gesture_map'].keys())
    
    # Print classification report
    print("\nClassification Report:")
    print(classification_report(labels, predictions, target_names=classes))
    
    # Plot confusion matrix
    plot_confusion_matrix(labels, predictions, classes)
    
    # Print model size information
    size_info = calculate_model_size(model)
    print("\nModel Resource Requirements:")
    print(f"Parameters: {size_info['parameter_count']:,}")
    print(f"Model size: {size_info['model_size_kb']:.2f} KB")
    print(f"Runtime RAM: {size_info['runtime_memory_kb']:.2f} KB")

if __name__ == "__main__":
    main()

