# Radar-Based Hand Gesture Recognition

## Overview
This repository implements a deep learning solution for hand gesture recognition using radar sensor data. The system processes raw radar signals from the BGT60TR13C XENSIV™ 60GHz radar sensor to recognize five distinct hand gestures (Push, SwipeDown, SwipeLeft, SwipeRight, SwipeUp) and background motion.

### Key Features
- Raw radar data preprocessing and feature extraction
- Windowed gesture detection
- LSTM-based gesture classification
- Real-time inference capabilities
- Comprehensive evaluation tools

## Neural Network Architecture
The model uses a Long Short-Term Memory (LSTM) network architecture designed for temporal gesture recognition:

Key specifications:
- Input: Sequence of 20 frames, each with 5 features
- Hidden layers: 2-layer LSTM with 64 hidden units
- Dropout: 0.3 between LSTM layers
- Output: 6 classes (5 gestures + background)

Model size:
- Parameters: ~50K
- Model storage: ~200KB
- Runtime RAM: ~200KB

## Project Structure

- **src/**
  - `data_loader.py`: Data loading and preparation
  - `data_preprocessing.py`: Raw data processing pipeline
  - `model.py`: LSTM model definition
  - `train.py`: Training script
  - `evaluate.py`: Evaluation tools
  - `inference.py`: Real-time inference
- **utils/**
  - `visualization.py`: Visualization tools
- `setup.py`
- `requirements.txt`

# Installation  

## Prerequisites
- Python ≥3.10
- CUDA-capable GPU (optional, for faster training)

## Setup
1. Clone the repository:
   ```bash
   git clone https://github.com/uctopus/hand_gesture_recognition
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Download the raw dataset from here https://ieee-dataport.org/documents/60-ghz-fmcw-radar-gesture-dataset#files

4. Unzip the dataset and place the `full_data` folder in the data folder. Change the name of the folder to `raw`.

5. Run the preprocessing script:
   ```bash
   python src/data_preprocessing.py
   ```
6. Run the training script:
   ```bash
   python src/train.py
   ```
7. Run the evaluation script:
   ```bash
   python src/evaluate.py
   ```
8. You can visualize the results by running:
   ```bash
   python utils/visualization.py
   ```


## Raw dataset description
This dataset provides a collection of radar-based hand gesture recordings for human-computer interaction research. The data was collected using the **BGT60TR13C XENSIV™ 60GHz Frequency Modulated Continuous Radar sensor**, overcoming challenges faced by traditional optical gesture recognition methods such as lighting conditions and occlusions.

## Dataset Composition
The dataset consists of **25,000 nominal gestures** and **24,000 anomalous gestures**, recorded from multiple users in diverse environments. It is designed to support research on robustness and adaptability in gesture recognition.

### **Gesture Types**
- **Nominal Gestures**
  - Swipe Left
  - Swipe Right
  - Swipe Up
  - Swipe Down
  - Push
- **Anomalous Gestures** (performed by 8 users)
  - **Fast executions** (~0.1 sec, 1,000 samples)
  - **Slow executions** (~3 sec, 1,000 samples)
  - **Wrist executions** (performed with wrist movement instead of full arm, 1,000 samples)

## Data Collection Setup
- **Radar Configuration**
  - Frequency: **58.5 GHz - 62.5 GHz**
  - Range Resolution: **37.5mm**
  - Max Detection Range: **1.2 meters**
  - **32 chirps per burst** at **33 Hz frame rate**
  - Pulse Repetition Time: **300 µs**

- **Recording Conditions**
  - **8 individuals** performed gestures in **6 different indoor locations**:
    - e1: Closed-space meeting room
    - e2: Open-space office room
    - e3: Library
    - e4: Kitchen
    - e5: Exercise room
    - e6: Bedroom
  - Field of view: **±45°**
  - Distance to radar: **≤1 meter**
  - Gesture duration: **~0.5 sec (10 frames per gesture)**

## Data Format
Each gesture sequence is stored as a **NumPy (.npy) file** with a **4D array (100x3x32x64)**:
- **100**: Frame length
- **3**: Number of virtual antennas
- **32**: Chirps per frame
- **64**: Samples per chirp

### **File Naming Convention**
- **Nominal Data:** `GestureName_EnvironmentLabel_UserLabel_SampleLabel.npy`
- **Anomalous Data:** `GestureName_AnomalyLabel_EnvironmentLabel_UserLabel_SampleLabel.npy`

### **User Labels**
- p1: Male
- p2: Female
- p3: Female
- p4: Male
- p5: Male
- p6: Male
- p7: Male
- p8: Male
- p9: Male
- p10: Female
- p11: Male
- p12: Male

## Dataset Files
| Dataset Name                | Size  |
|-----------------------------|-------|
| `fulldata_zipped.zip`       | 48.1GB |
| `fullextended_data.zip`     | 46.8GB |

## Extended Dataset Updates
- Added **4,000 nominal gestures** from **4 new users (p9, p10, p11, p12)** in location e1.
- **Anomalous gestures** collected from **8 users (p1, p2, p6, p7, p9, p10, p11, p12)**.
- **Same radar setup and gesture types** as the original dataset.

## Applications
This dataset is useful for:
- **Machine learning-based gesture recognition**
- **Human-computer interaction research**
- **Radar-based motion sensing applications**
- **Gesture-based control in robotics and smart devices**

---
This dataset provides a diverse and challenging benchmark for gesture recognition models, enabling robust performance evaluation in real-world scenarios.

### Data Processing Pipeline
1. **Raw Data**: 
   - Format: 100 frames × 3 antennas × 32 chirps × 64 samples
   - Contains raw radar signal data

2. **Processed Features**:
   - 5 key features extracted per frame:
     - Radial Distance
     - Radial Velocity
     - Horizontal Angle
     - Vertical Angle
     - Signal Magnitude

3. **Windowed Segments**:
   - 20-frame windows containing gesture movements
   - Automatically extracted from processed features
   - Background segments for non-gesture motion

## Results and Evaluation

### Metrics
- **Accuracy**: Overall classification accuracy across all gesture types
- **Precision**: Ability to avoid false positive gesture detections
- **Recall**: Ability to detect all instances of each gesture
- **F1-score**: Harmonic mean of precision and recall
- **Confusion Matrix**: Detailed view of classification performance per gesture

### Performance
Our LSTM model achieves the following metrics on the test set:

| Gesture     | Precision | Recall | F1-Score | Support |
|-------------|-----------|--------|----------|---------|
| Push        | 0.98      | 0.97   | 0.97     | 829     |
| SwipeDown   | 0.96      | 0.95   | 0.95     | 690     |
| SwipeLeft   | 0.97      | 0.96   | 0.96     | 731     |
| SwipeRight  | 0.97      | 0.98   | 0.97     | 746     |
| SwipeUp     | 0.95      | 0.96   | 0.95     | 686     |
| Background  | 0.98      | 0.98   | 0.98     | 2059    |

**Overall Results:**
- Average Accuracy: 97%
- Macro Avg F1-Score: 0.96
- Weighted Avg F1-Score: 0.97

### Key Findings
1. **Robust Gesture Detection**: High accuracy across all gesture types
2. **Background Discrimination**: Excellent performance in distinguishing gestures from background motion
3. **User Independence**: Consistent performance across different users and environments
4. **Real-time Capability**: Inference time < 10ms on CPU, suitable for real-time applications

### Model Characteristics
- **Training Time**: ~45 minutes on older CPU Intel Core i7 8th gen
- **Convergence**: Typically reaches 95% validation accuracy within 3 epochs
- **Memory Efficiency**: Small model footprint suitable for embedded systems