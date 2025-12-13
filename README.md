# Sentiment Classification with Neural Networks

This project applies convolutional neural networks (CNNs) to classify tweet sentiment using the Sentiment140 dataset. The goal is to evaluate how changes in architecture and hyperparameters affect accuracy, convergence behavior, and computational cost in large-scale text classification.

## Dataset
The Sentiment140 dataset contains 1.6 million tweets labeled as positive or negative. After balancing the dataset, it was split into 80% training, 10% validation, and 10% testing.

## Models

### Baseline CNN
- 100-dimensional embedding layer  
- 1D convolution with 128 filters (kernel size 5)  
- Global max pooling  
- Sigmoid output layer  
- Total parameters: 1,064,357  

### Modified CNN (Architecture Experiment)
- Adds one additional Conv1D layer
- Wider filter bank  
- Slightly larger kernel size in early layers  
- Total parameters: 1,507,749  

A separate modified CNN was used for the hyperparameter experiments, allowing adjustments to kernel size, embedding dimension, and classifier depth.

## Training Setup
All training used:
- Batch size = 32  
- Maximum of 20 epochs  
- Early stopping (patience = 3)

Learning rates of 0.001, 0.01, and 0.1 were used for the architecture experiment.  
The hyperparameter experiment additionally varied optimizer, hidden layers, kernel size, and embedding dimension.

## Architecture Experiment Results

### Test Accuracy
| Model | Learning Rate | Test Accuracy |
|-------|---------------|---------------|
| Baseline CNN | 0.001 | 76.81% |
| Modified CNN | 0.001 | 76.19% |
| Baseline CNN | 0.01  | 77.62% |
| Modified CNN | 0.01  | 77.14% |
| Baseline CNN | 0.1   | 78.14% |
| Modified CNN | 0.1   | 77.59% |

The baseline CNN outperformed the deeper modified CNN across all learning rates.

## Hyperparameter Experiment Results

The hyperparameter study tested different learning rates, optimizers, kernel sizes, embedding sizes, and classifier depths.

### Best Accuracy per Setting
| Setting | Value | Test Accuracy |
|---------|--------|----------------|
| Learning rate | 0.1 | 79.07% |
| Optimizer | SGD | 78.58% |
| Hidden layers | 3 | 78.63% |
| Input length | 50 | 78.73% |
| Kernel size | 3 | 79.12% |
| Embedding dimension | 500 | 79.54% |

The strongest performance overall was obtained using a 500-dimensional embedding.

## Key Findings
- The baseline CNN generalized better than the deeper architecture despite having fewer parameters.  
- Architectural depth did not yield accuracy gains and sometimes introduced mild overfitting.  
- Hyperparameters had a far greater impact on performance than added model complexity.  
- Higher learning rates, larger embeddings, smaller kernels, and shorter input lengths consistently improved results.  
- Adam diverged at the learning rates used; future work should test it at smaller learning rates for a fair comparison.  
- The best overall accuracy (79.54%) was achieved using a modified CNN with a 500-dimensional embedding.

## Author
Saisrijith Reddy Maramreddy

