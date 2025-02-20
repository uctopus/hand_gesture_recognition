import torch
import os
from model import LSTMModel
from data_loader import GestureDataLoader

class GesturePredictor:
    def __init__(self, model_path='models/final_model.pth'):
        """
        Initialize the gesture predictor
        
        Args:
            model_path: Path to the trained model file
        """
        if not os.path.exists(model_path):
            raise FileNotFoundError(
                f"Model file not found at {model_path}. "
                "Please train the model first using train.py"
            )
            
        # Load model and gesture map
        checkpoint = torch.load(model_path)
        
        # Check if it's a state dict or full checkpoint
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            self.gesture_map = checkpoint['gesture_map']
            model_state = checkpoint['model_state_dict']
        else:
            # If it's just the state dict (best_model.pth)
            self.gesture_map = {
                'Push': 0,
                'SwipeDown': 1,
                'SwipeLeft': 2,
                'SwipeRight': 3,
                'SwipeUp': 4,
                'Background': 5
            }
            model_state = checkpoint
            
        self.gesture_map_inv = {v: k for k, v in self.gesture_map.items()}
        
        # Initialize model
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = LSTMModel().to(self.device)
        self.model.load_state_dict(model_state)
        self.model.eval()
    
    def predict(self, features):
        """
        Predict gesture from features
        
        Args:
            features: numpy array of shape (20, 5) - windowed gesture features
            
        Returns:
            str: Predicted gesture name
            float: Confidence score
        """
        # Prepare input
        x = torch.FloatTensor(features).unsqueeze(0).to(self.device)
        
        # Get prediction
        with torch.no_grad():
            outputs = self.model(x)
            probabilities = torch.softmax(outputs, dim=1)
            confidence, predicted = torch.max(probabilities, 1)
        
        # Get gesture name
        gesture = self.gesture_map_inv[predicted.item()]
        
        return gesture, confidence.item()

def main():
    # Check if models directory exists
    if not os.path.exists('models'):
        print("\nError: 'models' directory not found.")
        print("Please train the model first using train.py")
        return
        
    # Try different model files
    model_files = ['lstm_model.pth']
    predictor = None
    
    for model_file in model_files:
        model_path = os.path.join('models', model_file)
        try:
            predictor = GesturePredictor(model_path)
            print(f"\nLoaded model from: {model_path}")
            break
        except FileNotFoundError:
            continue
    
    if predictor is None:
        print("\nError: No trained model found.")
        print("Please train the model first using train.py")
        return
    
    # Load a test gesture from processed directory
    data_loader = GestureDataLoader(
        raw_data_dir='data/raw',
        processed_data_dir='data/processed',
        windowed_data_dir='data/windowed'
    )
    
    # Get prediction for first test file
    test_files = data_loader.get_windowed_gesture_files()
    
    if not test_files:
        print("\nError: No windowed gesture files found.")
        print("Please process the raw data first using data_preprocessing.py")
        return
    
    # Get prediction for first test file
    test_file = test_files[0]
    features = data_loader.load_windowed_gesture(test_file)
    
    # Get prediction
    gesture, confidence = predictor.predict(features)
    print(f"\nTest prediction:")
    print(f"File: {test_file}")
    print(f"Predicted gesture: {gesture}")
    print(f"Confidence: {confidence:.2f}")

if __name__ == "__main__":
    main()

