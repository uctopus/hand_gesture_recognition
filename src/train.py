import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
from pathlib import Path

from model import LSTMModel, GestureDataset
from data_loader import GestureDataLoader

def train_model(model, train_loader, val_loader, criterion, optimizer, 
                num_epochs=30, device='cuda'):
    """Train the model and return training history"""
    best_val_loss = float('inf')
    train_losses = []
    val_losses = []
    
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        train_loss = 0
        for features, labels in train_loader:
            features, labels = features.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(features)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
        
        train_loss /= len(train_loader)
        train_losses.append(train_loss)
        
        # Validation phase
        model.eval()
        val_loss = 0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for features, labels in val_loader:
                features, labels = features.to(device), labels.to(device)
                outputs = model(features)
                loss = criterion(outputs, labels)
                
                val_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        val_loss /= len(val_loader)
        val_losses.append(val_loss)
        
        accuracy = 100 * correct / total
        print(f'Epoch [{epoch+1}/{num_epochs}], '
              f'Train Loss: {train_loss:.4f}, '
              f'Val Loss: {val_loss:.4f}, '
              f'Val Accuracy: {accuracy:.2f}%')
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), 'models/best_model.pth')
    
    return train_losses, val_losses

def main():
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create directories if they don't exist
    Path('models').mkdir(exist_ok=True)
    Path('data/processed').mkdir(exist_ok=True, parents=True)
    
    # Load and prepare data
    data_loader = GestureDataLoader()
    data = data_loader.prepare_data(max_background_ratio=3)
    
    # Create datasets
    train_dataset = GestureDataset(*data['train'])
    val_dataset = GestureDataset(*data['val'])
    test_dataset = GestureDataset(*data['test'])
    
    # Create dataloaders
    batch_size = 32
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)
    
    # Initialize model
    model = LSTMModel(
        input_size=5,
        hidden_size=64,
        num_layers=2,
        num_classes=6
    ).to(device)
    
    # Calculate class weights for balanced training
    class_counts = np.bincount(data['train'][1])
    class_weights = torch.FloatTensor(1.0 / class_counts).to(device)
    
    # Define loss and optimizer
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # Train model
    print("\nTraining model...")
    train_losses, val_losses = train_model(
        model, train_loader, val_loader, criterion, optimizer, device=device
    )
    
    # Save final model
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'train_losses': train_losses,
        'val_losses': val_losses,
        'gesture_map': data['gesture_map']
    }, 'models/final_model.pth')

if __name__ == "__main__":
    main()



