import torch
import torch.nn as nn
import numpy as np

class LSTMModel(nn.Module):
    def __init__(self, input_size=5, hidden_size=64, num_layers=2, num_classes=6):
        super(LSTMModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, 
                           batch_first=True, dropout=0.3 if num_layers > 1 else 0)
        self.fc = nn.Linear(hidden_size, num_classes)
        
    def forward(self, x):
        # Initialize hidden state with zeros
        device = x.device
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)
        
        # Forward propagate LSTM
        out, _ = self.lstm(x, (h0, c0))
        
        # Decode the hidden state of the last time step
        out = self.fc(out[:, -1, :])
        return out

class GestureDataset(torch.utils.data.Dataset):
    def __init__(self, features, labels):
        self.features = torch.FloatTensor(features)
        self.labels = torch.LongTensor(labels)
    
    def __len__(self):
        return len(self.features)
    
    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]

def calculate_model_size(model):
    """
    Calculate model size and memory requirements
    
    Args:
        model: PyTorch model instance
        
    Returns:
        dict containing size calculations
    """
    # Count parameters
    param_size = 0
    param_count = 0
    
    for param in model.parameters():
        param_count += param.numel()
        # Each parameter is stored as float32 (4 bytes)
        param_size += param.numel() * 4
    
    # Calculate buffer size (for batch norm, etc.)
    buffer_size = 0
    for buffer in model.buffers():
        buffer_size += buffer.numel() * 4
    
    # Calculate model size in different units
    total_size = param_size + buffer_size
    size_kb = total_size / 1024
    size_mb = size_kb / 1024
    
    # Estimate runtime RAM for inference
    # Consider input size, intermediate activations, and output
    batch_size = 1
    input_size = (batch_size, 20, 5)  # (batch, sequence, features)
    input_memory = np.prod(input_size) * 4  # float32
    
    # LSTM memory: hidden states, cell states, gates
    lstm_memory = (
        2 * model.num_layers * batch_size * model.hidden_size * 4 +  # hidden & cell states
        4 * batch_size * model.hidden_size * 4  # gates
    )
    
    # Output memory
    output_memory = batch_size * 6 * 4  # 6 classes, float32
    
    total_runtime_memory = input_memory + lstm_memory + output_memory + total_size
    
    return {
        'parameter_count': param_count,
        'model_size_bytes': total_size,
        'model_size_kb': size_kb,
        'model_size_mb': size_mb,
        'runtime_memory_kb': total_runtime_memory / 1024,
        'runtime_memory_mb': total_runtime_memory / 1024 / 1024
    }

if __name__ == "__main__":
    # Test model architecture and calculate size
    model = LSTMModel(
        input_size=5,
        hidden_size=64,
        num_layers=2,
        num_classes=6
    )
    
    # Print model architecture
    print("\nModel Architecture:")
    print(model)
    
    # Calculate and print model size
    size_info = calculate_model_size(model)
    
    print("\nModel Size Information:")
    print(f"Number of parameters: {size_info['parameter_count']:,}")
    print(f"Model size: {size_info['model_size_kb']:.2f} KB ({size_info['model_size_mb']:.2f} MB)")
    print("\nMemory Requirements:")
    print(f"Flash memory (model storage): {size_info['model_size_kb']:.2f} KB")
    print(f"RAM for inference: {size_info['runtime_memory_kb']:.2f} KB "
          f"({size_info['runtime_memory_mb']:.2f} MB)")
    
    # Print layer-by-layer parameters
    print("\nParameters by layer:")
    for name, param in model.named_parameters():
        print(f"{name}: {param.numel():,} parameters")

