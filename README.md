# Facial Expression Recognition System

This project implements a real-time facial expression recognition system using deep learning and computer vision. It can detect and classify seven basic emotions: Angry, Disgust, Fear, Happy, Sad, Surprise, and Neutral.

## Requirements

- Python 3.8 or higher
- Webcam
- Required Python packages (listed in requirements.txt)

## Installation

1. Clone this repository
2. Install the required packages:
```bash
pip install -r requirements.txt
```

## Dataset

The model is trained on the FER2013 dataset. You need to download and organize the dataset in the following structure:

```
data/
├── train/
│   ├── angry/
│   ├── disgust/
│   ├── fear/
│   ├── happy/
│   ├── sad/
│   ├── surprise/
│   └── neutral/
└── test/
    ├── angry/
    ├── disgust/
    ├── fear/
    ├── happy/
    ├── sad/
    ├── surprise/
    └── neutral/
```

## Training the Model

To train the model, run:
```bash
python train_model.py
```

This will:
1. Train the model on the dataset
2. Save the trained model as 'model.h5'
3. Generate accuracy and loss plots

## Running the Application

To run the facial expression recognition system:
```bash
python app.py
```

The application will:
1. Open your webcam
2. Detect faces in real-time
3. Classify the facial expressions
4. Display the results on the screen

Press 'q' to quit the application.

## Features

- Real-time facial expression detection
- Seven emotion classification
- Confidence score display
- Face detection using Haar Cascade
- Deep learning-based emotion recognition

## Model Architecture

The model uses a Convolutional Neural Network (CNN) with the following layers:
- Multiple Conv2D layers with ReLU activation
- MaxPooling2D layers
- Dropout layers for regularization
- Dense layers for classification

## Performance

The model achieves good accuracy on the FER2013 dataset. Training progress can be monitored through the generated plots:
- accuracy_plot.png
- loss_plot.png 
