import numpy as np
import matplotlib.pyplot as plt
import sys
import os


def plot_gesture_features(features_norm, filename):
    """
    Plot normalized features from a single gesture
    
    Args:
        features_norm: numpy array of shape (n_frames, 5) containing normalized features
        filename: source filename for plot title
    """
    n_frames = features_norm.shape[0]
    time_norm = np.linspace(0, 1, n_frames)
    
    fig, axes = plt.subplots(5, 1, figsize=(12, 15))
    plt.subplots_adjust(hspace=0.4)
    
    feature_names = ['Radial Distance', 'Radial Velocity', 
                    'Horizontal Angle', 'Vertical Angle', 
                    'Signal Magnitude']
    
    for i, name in enumerate(feature_names):
        axes[i].plot(time_norm, features_norm[:, i])
        axes[i].set_title(name)
        axes[i].set_xlabel('Normalized Time')
        axes[i].set_ylabel('Normalized Value')
        axes[i].grid(True)
    
    plt.suptitle(f'Normalized Features from {filename.split("/")[-1]}')
    plt.show()

def plot_gesture_statistics(gesture_files_by_type, features_data):
    """
    Plot averaged features with standard deviation for each gesture type
    
    Args:
        gesture_files_by_type: dictionary mapping gesture types to file lists
        features_data: dictionary containing mean and std for each gesture type
    """
    gesture_types = list(gesture_files_by_type.keys())
    feature_names = ['Radial Distance', 'Radial Velocity', 
                    'Horizontal Angle', 'Vertical Angle', 
                    'Signal Magnitude']
    
    # Create subplot grid
    fig, axes = plt.subplots(5, len(gesture_types), figsize=(15, 15))
    plt.subplots_adjust(hspace=0.4, wspace=0.3)
    
    x_new = np.linspace(0, 1, 100)  # Standard time points
    
    for col, gesture in enumerate(gesture_types):
        mean_features = features_data[gesture]['mean']
        std_features = features_data[gesture]['std']
        
        # Plot each feature
        for row in range(5):
            ax = axes[row, col]
            
            # Plot std deviation area
            ax.fill_between(x_new, 
                          mean_features[:, row] - std_features[:, row],
                          mean_features[:, row] + std_features[:, row],
                          alpha=0.3, color='blue')
            
            # Plot mean line
            ax.plot(x_new, mean_features[:, row], 'k-', linewidth=2)
            
            # Formatting
            if row == 0:
                ax.set_title(gesture)
            if col == 0:
                ax.set_ylabel(feature_names[row])
            if row == 4:
                ax.set_xlabel('time')
            
            ax.grid(True)
            
            # Remove x-ticks except for bottom row
            if row != 4:
                ax.set_xticks([])
    
    plt.suptitle('Gesture Feature Characteristics\n(mean ± std)', y=1.02)
    plt.show()

def print_feature_statistics(features):
    """
    Print statistics for normalized features
    
    Args:
        features: numpy array of shape (n_frames, 5) containing normalized features
    """
    feature_names = ['Radial Distance', 'Radial Velocity', 
                    'Horizontal Angle', 'Vertical Angle', 
                    'Signal Magnitude']
    
    print("\nFeature Statistics:")
    for i, name in enumerate(feature_names):
        values = features[:, i]
        print(f"\n{name}:")
        print(f"  Mean: {np.mean(values):.3f}")
        print(f"  Std:  {np.std(values):.3f}")
        print(f"  Min:  {np.min(values):.3f}")
        print(f"  Max:  {np.max(values):.3f}")

if __name__ == "__main__":
    # Add the project root directory to Python path
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    sys.path.append(project_root)
    
    # Now we can import from src
    from src.data_loader import GestureDataLoader
    
    # Initialize data loader
    data_loader = GestureDataLoader()
    
    # 1. Plot processed gesture features
    print("\nPlotting processed gesture features...")
    processed_file = "data/processed/processed_Push_e1_u1_s1.npy"
    if os.path.exists(processed_file):
        features = data_loader.load_processed_gesture(processed_file)
        plot_gesture_features(features, processed_file)
    
    # 2. Plot gesture statistics
    print("\nPlotting gesture statistics...")
    gesture_files = data_loader.get_gesture_files_by_type(data_loader.processed_data_dir)
    if gesture_files:
        features_data = data_loader.calculate_gesture_statistics(gesture_files)
        plot_gesture_statistics(gesture_files, features_data) 