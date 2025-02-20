import numpy as np
from scipy.signal import windows
import os
from data_loader import GestureDataLoader

class GestureProcessor:
    """
    Class for processing radar gesture data.
    Handles feature extraction and normalization from raw radar frames.
    """
    def __init__(self, norm_mean, norm_scale, min_range_bin=3):
        """
        Initialize the gesture processor.
        
        Args:
            norm_mean: List of mean values for normalizing features
            norm_scale: List of scale values for normalizing features
            min_range_bin: Minimum range bin to consider (default=3)
        """
        self.norm_mean = norm_mean
        self.norm_scale = norm_scale
        self.min_range_bin = min_range_bin
        
        # Constants for radar configuration
        self.n_range_bins = 32
        self.doppler_window_beta = 2.5
        self.azimuth_offset = np.deg2rad(8.0)
        self.elevation_offset = np.deg2rad(24.0)
    
    def slim_algo(self, frame):
        """
        Process single radar frame to extract features for gesture recognition.
        
        Args:
            frame: numpy array of shape (3, 32, 64) - single frame data
            
        Returns:
            dict: Dictionary containing extracted features
        """
        n_channels, n_chirps, n_samples = frame.shape
        
        # 1. Range Processing
        range_window = windows.hann(n_samples)
        frame_windowed = frame * range_window[None, None, :]
        
        # Range FFT
        range_fft = np.fft.fft(frame_windowed, axis=2)
        range_fft = range_fft[:, :, :self.n_range_bins]  # Keep first half
        
        # Remove static targets
        range_fft = range_fft - np.mean(range_fft, axis=1, keepdims=True)
        
        # 2. Get Range Profile
        range_abs = np.abs(range_fft)
        range_abs_mean = np.mean(range_abs, axis=0)  # Average over channels
        
        # Calculate range profile
        range_profile = np.zeros(self.n_range_bins - self.min_range_bin)
        for idx_rb in range(self.min_range_bin, self.n_range_bins):
            chirp_sum = np.sum(range_abs_mean[1:, idx_rb])  # Skip first chirp
            range_profile[idx_rb - self.min_range_bin] = chirp_sum / (n_chirps - 1)
        
        # 3. Find peak in range profile
        idx_peak_range = np.argmax(range_profile) + self.min_range_bin
        
        # 4. Doppler Processing at peak range
        range_slice = range_fft[:, :, idx_peak_range]
        
        doppler_window = windows.kaiser(n_chirps, beta=self.doppler_window_beta)
        range_slice_windowed = range_slice * doppler_window[None, :]
        
        doppler_fft = np.fft.fftshift(np.fft.fft(range_slice_windowed, axis=1), axes=1)
        doppler_profile = np.mean(np.abs(doppler_fft), axis=0)
        
        idx_peak_doppler = np.argmax(doppler_profile)
        val_peak_doppler = doppler_profile[idx_peak_doppler]
        
        # 5. Angle Calculation using phase comparison monopulse
        peak_data = doppler_fft[:, idx_peak_doppler]
        azimuth = np.arcsin(np.angle(peak_data[2] / peak_data[0]) / np.pi) + self.azimuth_offset
        elevation = np.arcsin(np.angle(peak_data[2] / peak_data[1]) / np.pi) + self.elevation_offset
        
        return {
            'range_bin': idx_peak_range,
            'doppler_bin': idx_peak_doppler,
            'azimuth': azimuth,
            'elevation': elevation,
            'value': val_peak_doppler
        }
    
    def process_frame_normalized(self, frame):
        """
        Process single frame and normalize features.
        
        Args:
            frame: numpy array of shape (3, 32, 64) - single frame data
            
        Returns:
            numpy array: Normalized features vector of length 5
        """
        features = self.slim_algo(frame)
        return self._normalize_features(features)
    
    def _normalize_features(self, features):
        """
        Normalize extracted features using stored normalization parameters.
        
        Args:
            features: Dictionary of raw features
            
        Returns:
            numpy array: Normalized features vector of length 5
        """
        model_in = np.zeros(5)
        feature_keys = ['range_bin', 'doppler_bin', 'azimuth', 'elevation', 'value']
        
        for i, key in enumerate(feature_keys):
            model_in[i] = (float(features[key]) - self.norm_mean[i]) / self.norm_scale[i]
        
        return model_in
    
    def process_gesture_file(self, filename):
        """
        Process entire gesture file.
        
        Args:
            filename: Path to .npy gesture file
            
        Returns:
            numpy array: Array of shape (n_frames, 5) containing normalized features
        """
        data = np.load(filename)
        n_frames = data.shape[0]
        features = np.zeros((n_frames, 5))
        
        for i in range(n_frames):
            features[i] = self.process_frame_normalized(data[i])
        
        return features
    
    def process_and_save(self, input_file, output_file):
        """
        Process gesture file and save results.
        
        Args:
            input_file: Path to input .npy gesture file
            output_file: Path to save processed features
            
        Returns:
            numpy array: Processed features array
        """
        features = self.process_gesture_file(input_file)
        np.save(output_file, features)
        return features

    def process_all_gestures(self, data_loader, input_dir, output_dir):
        """
        Process all gesture files in the input directory using the data loader.
        
        Args:
            data_loader: GestureDataLoader instance
            input_dir: Directory containing raw gesture files
            output_dir: Directory to save processed features
            
        Returns:
            dict: Statistics about processed files
        """
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Get all gesture files using the data loader
        gesture_files = data_loader.get_gesture_files(input_dir)
        
        stats = {
            'total_files': len(gesture_files),
            'processed_files': 0,
            'errors': []
        }
        
        # Process each file
        for input_file in gesture_files:
            try:
                # Create output filename
                basename = os.path.basename(input_file)
                output_file = os.path.join(output_dir, f"processed_{basename}")
                
                # Process and save
                self.process_and_save(input_file, output_file)
                stats['processed_files'] += 1
                
            except Exception as e:
                stats['errors'].append({
                    'file': input_file,
                    'error': str(e)
                })
        
        return stats

    def find_gesture_window(self, features, signal_threshold=0.5, window_size=20):
        """
        Find the gesture window within the feature sequence based on signal magnitude and radial distance.
        
        Args:
            features: numpy array of shape (n_frames, 5) containing normalized features
            signal_threshold: minimum signal magnitude to consider
            window_size: number of frames to include in gesture window
        
        Returns:
            start_idx, end_idx: indices of gesture window
            is_valid: whether a valid gesture was found
        """
        radial_distance = features[:, 0]
        signal_magnitude = features[:, 4]
        
        # Find frames with sufficient signal strength
        valid_frames = signal_magnitude > signal_threshold
        
        if not np.any(valid_frames):
            return 0, 0, False
        
        # Find frame with minimum radial distance among valid frames
        valid_indices = np.where(valid_frames)[0]
        min_distance_idx = valid_indices[np.argmin(radial_distance[valid_frames])]
        
        # Calculate window boundaries
        half_window = window_size // 2
        start_idx = max(0, min_distance_idx - half_window)
        end_idx = min(len(features), start_idx + window_size)
        
        # Adjust start if end was capped
        if end_idx - start_idx < window_size:
            start_idx = max(0, end_idx - window_size)
        
        # Verify average signal magnitude in the window
        window_signal = np.mean(signal_magnitude[start_idx:end_idx])
        is_valid = window_signal > signal_threshold
        
        return start_idx, end_idx, is_valid

    def extract_windows(self, features, window_size=20):
        """
        Extract gesture and background windows from features.
        
        Args:
            features: numpy array of shape (n_frames, 5)
            window_size: size of the windows to extract
            
        Returns:
            dict containing gesture and background windows
        """
        start_idx, end_idx, is_valid = self.find_gesture_window(features, window_size=window_size)
        
        if not is_valid:
            return None
            
        # Extract gesture window
        gesture_window = features[start_idx:end_idx]
        
        # Extract background windows
        background_windows = []
        
        # Before gesture
        if start_idx >= window_size:
            for i in range(0, start_idx - window_size + 1, window_size):
                bg_window = features[i:i + window_size]
                if len(bg_window) == window_size:
                    background_windows.append(bg_window)
        
        # After gesture
        remaining_frames = len(features) - end_idx
        if remaining_frames >= window_size:
            for i in range(end_idx, len(features) - window_size + 1, window_size):
                bg_window = features[i:i + window_size]
                if len(bg_window) == window_size:
                    background_windows.append(bg_window)
        
        return {
            'gesture': gesture_window,
            'background': background_windows,
            'window_indices': (start_idx, end_idx)
        }

    def process_and_save_windows(self, input_file, output_dir, window_size=20, signal_threshold=0.5):
        """
        Process a gesture file and save windowed segments.
        
        Args:
            input_file: Path to processed gesture file
            output_dir: Directory to save windowed segments
            window_size: Size of windows to extract
            signal_threshold: Threshold for valid gesture detection
        
        Returns:
            dict: Information about extracted windows
        """
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Load and extract windows
        features = self.process_gesture_file(input_file)
        windows = self.extract_windows(features, window_size)
        
        if not windows:
            return {'status': 'no_valid_gesture'}
        
        # Get base filename
        basename = os.path.splitext(os.path.basename(input_file))[0]
        
        # Save gesture window
        gesture_file = os.path.join(output_dir, f"windowed_{basename}.npy")
        np.save(gesture_file, windows['gesture'])
        
        # Save background windows
        bg_files = []
        for i, bg_window in enumerate(windows['background']):
            bg_file = os.path.join(output_dir, 
                f"background_{basename}_{'pre' if i < len(windows['background'])/2 else 'post'}_{i}.npy")
            np.save(bg_file, bg_window)
            bg_files.append(bg_file)
        
        return {
            'status': 'success',
            'gesture_file': gesture_file,
            'background_files': bg_files,
            'window_indices': windows['window_indices']
        }

    def process_all_windows(self, data_loader, input_dir, output_dir):
        """
        Process all gesture files and create windowed segments.
        
        Args:
            data_loader: GestureDataLoader instance
            input_dir: Directory containing processed gesture files
            output_dir: Directory to save windowed segments
        """
        # Get all processed files
        processed_files = data_loader.get_gesture_files(input_dir)
        
        stats = {
            'total_files': len(processed_files),
            'successful': 0,
            'no_gesture': 0,
            'errors': []
        }
        
        for file in processed_files:
            try:
                result = self.process_and_save_windows(file, output_dir)
                
                if result['status'] == 'success':
                    stats['successful'] += 1
                else:
                    stats['no_gesture'] += 1
                    
            except Exception as e:
                stats['errors'].append({
                    'file': file,
                    'error': str(e)
                })
        
        return stats

# Default normalization parameters
DEFAULT_NORM_MEAN = [9.26814552650607, 4.391583164927378, 0.27332462978312866,
                    -0.02838213175529301, 0.00026668613549266876]
DEFAULT_NORM_SCALE = [5.801363069954616, 7.547439540930497, 0.5629401789624862,
                     0.41502512890635995, 3000]

if __name__ == "__main__":
    # Initialize processor and data loader
    processor = GestureProcessor(DEFAULT_NORM_MEAN, DEFAULT_NORM_SCALE)
    data_loader = GestureDataLoader()
    
    # Process raw data to normalized features
    input_dir = "data/raw"
    processed_dir = "data/processed"
    windowed_dir = "data/windowed"
    
    # Step 1: Process raw files
    print("\nStep 1: Processing raw files...")
    stats = processor.process_all_gestures(data_loader, input_dir, processed_dir)
    print(f"Processed {stats['processed_files']} of {stats['total_files']} files")
    
    # Step 2: Create windowed segments
    print("\nStep 2: Creating windowed segments...")
    window_stats = processor.process_all_windows(data_loader, processed_dir, windowed_dir)
    print(f"Created windows for {window_stats['successful']} files")
    print(f"No valid gesture in {window_stats['no_gesture']} files")
    print(f"Errors in {len(window_stats['errors'])} files")
