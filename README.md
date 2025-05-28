# MNIST Drawing Interface

An interactive drawing application that allows users to draw numbers on a 28x28 grid and uses a Convolutional Neural Network (CNN) to predict the drawn digit, similar to the MNIST dataset.

## Description

This project creates a user-friendly interface where users can draw numbers using their mouse or touch input. The drawing is processed through a CNN model trained on the MNIST dataset to predict the drawn digit. The interface provides real-time feedback and predictions.

## Features

- Interactive 28x28 drawing canvas
- Real-time digit prediction
- Clean and intuitive user interface
- Support for mouse and touch input
- MNIST-style digit recognition

## Requirements

- Python 3.x
- NumPy
- Pygame
- Machine Learning library (TBD)
- Additional dependencies will be listed in requirements.txt

## Installation

1. Clone this repository:
```bash
git clone [repository-url]
cd MNIST_PLAYGROUND
```

2. Install the required dependencies:
```bash
pip install -r requirements.txt
```

## Usage

1. Run the main application:
```bash
python main.py
```

2. Draw a number on the 28x28 grid using your mouse or touch input
3. The application will predict the drawn digit in real-time

## Project Structure

```
MNIST_PLAYGROUND/
├── main.py              # Main application entry point
├── model/              # CNN model implementation
├── interface/          # Drawing interface implementation
├── utils/             # Utility functions
└── requirements.txt    # Project dependencies
```

## Contributing

Feel free to submit issues and enhancement requests!

## License

[License information to be added]

## Acknowledgments

- MNIST dataset
- Pygame community
- NumPy community
