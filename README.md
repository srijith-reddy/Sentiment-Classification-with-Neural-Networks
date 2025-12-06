# Sentiment Classification with Neural Networks

This project applies convolutional neural networks (CNNs) to classify tweet sentiment using the Sentiment140 dataset. The goal is to evaluate how model architecture and learning rate choices affect performance on large-scale, informal text data.

## Overview
The Sentiment140 dataset contains 1.6M tweets labeled as positive or negative. After balancing the dataset, it was split into 80% training, 10% validation, and 10% testing.

Two CNN architectures were implemented:

### Baseline CNN
- 100-dimensional embedding layer  
- 1D convolution (128 filters, kernel size 5)  
- Global max pooling  
- Sigmoid output  

### Modified CNN
- Adds a second Conv1D layer with 256 filters  
- Uses a larger kernel size (7)  
- Increases parameter count by roughly 40 percent  

Both models were trained using SGD (batch size 32) with learning rates 0.1, 0.01, and 0.001.

## Results

### Test Accuracy Summary
| Model | Learning Rate | Test Accuracy |
|-------|----------------|---------------|
| Baseline CNN | 0.001 | 76.64% |
| Modified CNN | 0.001 | 76.74% |
| Baseline CNN | 0.01  | 77.39% |
| Modified CNN | 0.01  | 75.77% |
| Baseline CNN | 0.1   | 78.52% |
| Modified CNN | 0.1   | 78.13% |

### Parameter Count
- Baseline CNN: 1,064,357 parameters  
- Modified CNN: 1,507,749 parameters  

### Key Findings
- The baseline CNN performs as well as or slightly better than the modified model across learning rates.  
- A learning rate of 0.1 produced the strongest results.  
- The deeper, wider model did not generalize better despite its added capacity.  
- In this dataset, model simplicity provided more stable performance than added complexity.

## Author
Saisrijith Reddy Maramreddy
