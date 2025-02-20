import numpy as np
import glob
import os
from scipy.interpolate import interp1d
from sklearn.model_selection import train_test_split

class GestureDataLoader:
    def __init__(self, raw_data_dir='data/raw', 
                 processed_data_dir='data/processed',
                 windowed_data_dir='data/windowed'):
        self.raw_data_dir = raw_data_dir
        self.processed_data_dir = processed_data_dir
        self.windowed_data_dir = windowed_data_dir
        
    def load_raw_gesture(self, filename):
        """Load raw gesture data from .npy file"""
        return np.load(filename)
    
    def load_processed_gesture(self, filename):
        """Load processed gesture features"""
        return np.load(filename)
    
    def get_gesture_files(self, data_dir, pattern="*_e*_u*_s*.npy"):
        """Get all gesture files matching the pattern"""
        return glob.glob(os.path.join(data_dir, pattern))
    
    def get_gesture_files_by_type(self, data_dir):
        """Get files organized by gesture type"""
        gesture_types = ['Push', 'SwipeLeft', 'SwipeRight', 'SwipeUp', 'SwipeDown']
        files_by_type = {}
        
        for gesture in gesture_types:
            pattern = os.path.join(data_dir, f'processed_{gesture}_e*_u*_s*.npy')
            files_by_type[gesture] = glob.glob(pattern)
            
        return files_by_type
    
    def load_and_resample_features(self, filename, target_length=100):
        """Load features and resample to target length"""
        features = np.load(filename)
        n_frames = features.shape[0]
        
        # Resample to fixed length
        x_old = np.linspace(0, 1, n_frames)
        x_new = np.linspace(0, 1, target_length)
        features_resampled = np.zeros((target_length, features.shape[1]))
        
        for j in range(features.shape[1]):
            f = interp1d(x_old, features[:, j])
            features_resampled[:, j] = f(x_new)
            
        return features_resampled
    
    def calculate_gesture_statistics(self, files_by_type):
        """Calculate mean and std for each gesture type"""
        stats = {}
        
        for gesture, files in files_by_type.items():
            all_features = []
            
            for file in files:
                features = self.load_and_resample_features(file)
                all_features.append(features)
            
            all_features = np.array(all_features)
            stats[gesture] = {
                'mean': np.mean(all_features, axis=0),
                'std': np.std(all_features, axis=0)
            }
            
        return stats
    
    def get_windowed_gesture_files(self):
        """Get all windowed gesture files"""
        return glob.glob(os.path.join(self.windowed_data_dir, "windowed_*.npy"))
    
    def get_background_files(self):
        """Get all background segment files"""
        return glob.glob(os.path.join(self.windowed_data_dir, "background_*.npy"))
    
    def load_windowed_gesture(self, filename):
        """Load windowed gesture data"""
        return np.load(filename)
    
    def load_background_segment(self, filename):
        """Load background segment data"""
        return np.load(filename)
    
    def get_windowed_files_by_type(self):
        """Get windowed files organized by gesture type"""
        gesture_types = ['Push', 'SwipeLeft', 'SwipeRight', 'SwipeUp', 'SwipeDown']
        files_by_type = {}
        
        for gesture in gesture_types:
            pattern = os.path.join(self.windowed_data_dir, f"windowed_processed_{gesture}_*.npy")
            files_by_type[gesture] = glob.glob(pattern)
            
        return files_by_type
    
    def prepare_data(self, max_background_ratio=3):
        """
        Load and split gesture data with controlled background ratio
        
        Args:
            max_background_ratio: Maximum ratio of background to gesture samples
        """
        gesture_map = {
            'Push': 0,
            'SwipeDown': 1,
            'SwipeLeft': 2,
            'SwipeRight': 3,
            'SwipeUp': 4,
            'Background': 5
        }
        
        # First, organize files by user to prevent data leakage
        user_files = {}
        
        for gesture in gesture_map.keys():
            if gesture == 'Background':
                files = self.get_background_files()
            else:
                files = glob.glob(os.path.join(self.windowed_data_dir, 
                                             f"windowed_processed_{gesture}_*.npy"))
            
            # Group files by user
            for file in files:
                # Extract user ID from filename (e.g., "gesture_e1_u2_s123.npy" -> "u2")
                user_id = file.split('_u')[1].split('_')[0]
                if user_id not in user_files:
                    user_files[user_id] = []
                user_files[user_id].append((file, gesture))
        
        # Split users into train, validation, and test sets
        user_ids = list(user_files.keys())
        np.random.shuffle(user_ids)
        
        n_users = len(user_ids)
        n_test = max(1, int(0.2 * n_users))
        n_val = max(1, int(0.2 * (n_users - n_test)))
        
        test_users = user_ids[:n_test]
        val_users = user_ids[n_test:n_test + n_val]
        train_users = user_ids[n_test + n_val:]
        
        # Function to process files for a set of users
        def process_user_files(user_ids):
            features_by_class = {gesture: [] for gesture in gesture_map.keys()}
            
            for user_id in user_ids:
                for file, gesture in user_files[user_id]:
                    features = np.load(file)
                    if features.shape == (20, 5):
                        features_by_class[gesture].append(features)
            
            # Balance background samples
            if 'Background' in features_by_class:
                min_gesture_count = min(len(features_by_class[g]) 
                                      for g in gesture_map.keys() if g != 'Background')
                target_bg_count = min(len(features_by_class['Background']),
                                    min_gesture_count * max_background_ratio)
                
                if len(features_by_class['Background']) > target_bg_count:
                    indices = np.random.choice(
                        len(features_by_class['Background']), 
                        target_bg_count, 
                        replace=False
                    )
                    features_by_class['Background'] = [features_by_class['Background'][i] 
                                                     for i in indices]
            
            # Combine features and labels
            X, y = [], []
            for gesture, features_list in features_by_class.items():
                X.extend(features_list)
                y.extend([gesture_map[gesture]] * len(features_list))
            
            return np.array(X), np.array(y)
        
        # Process each split
        X_train, y_train = process_user_files(train_users)
        X_val, y_val = process_user_files(val_users)
        X_test, y_test = process_user_files(test_users)
        
        print("\nData split statistics:")
        print(f"Training users: {len(train_users)}")
        print(f"Validation users: {len(val_users)}")
        print(f"Test users: {len(test_users)}")
        
        print("\nSample counts:")
        print(f"Training samples: {len(X_train)}")
        print(f"Validation samples: {len(X_val)}")
        print(f"Test samples: {len(X_test)}")
        
        return {
            'train': (X_train, y_train),
            'val': (X_val, y_val),
            'test': (X_test, y_test),
            'gesture_map': gesture_map
        }

if __name__ == "__main__":
    # Initialize data loader
    data_loader = GestureDataLoader()
    
    # Test all data loading functionalities
    print("\nTesting data loading capabilities:")
    
    # 1. Raw data
    raw_files = data_loader.get_gesture_files(data_loader.raw_data_dir)
    print(f"\n1. Raw data files found: {len(raw_files)}")
    
    # 2. Processed data
    processed_files = data_loader.get_gesture_files(data_loader.processed_data_dir)
    print(f"\n2. Processed data files found: {len(processed_files)}")
    
    # 3. Windowed data
    windowed_files = data_loader.get_windowed_gesture_files()
    background_files = data_loader.get_background_files()
    print(f"\n3. Windowed data:")
    print(f"   - Gesture segments: {len(windowed_files)}")
    print(f"   - Background segments: {len(background_files)}")
    
    # Test loading each type if files exist
    if raw_files:
        data = data_loader.load_raw_gesture(raw_files[0])
        print(f"\nRaw data shape: {data.shape}")
    
    if processed_files:
        data = data_loader.load_processed_gesture(processed_files[0])
        print(f"Processed data shape: {data.shape}")
    
    if windowed_files:
        data = data_loader.load_windowed_gesture(windowed_files[0])
        print(f"Windowed gesture shape: {data.shape}")
    
    if background_files:
        data = data_loader.load_background_segment(background_files[0])
        print(f"Background segment shape: {data.shape}")
