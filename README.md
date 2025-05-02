# Optimized Deep Convolutional Architectures for Cross-Domain Classification Tasks

## Project Overview

This project focuses on building and optimizing deep convolutional neural networks (CNNs) for robust classification across both image and audio domains. By integrating hybrid architectural blocks into custom ResNet, VGG, and Inception models, the system effectively handles diverse datasets such as **CIFAR-10** (image classification) and **SpeechCommands V0.02** (audio classification).

## Key Features

- **Custom CNN Architectures**: Implemented enhanced ResNet, VGG, and Inception variants, augmented with hybrid layers for improved feature representation and robustness.
- **Cross-Domain Generalization**: Trained models to achieve high accuracy on both visual and auditory datasets, enabling effective multi-modal learning.
- **Robust Training Pipeline**: Developed a flexible pipeline with custom checkpointing, enabling reproducibility, modular training, and efficient evaluation.
- **Benchmark-Driven Evaluation**: Models were optimized using **Cross-Entropy Loss** and the **Adam Optimizer**, reaching competitive benchmarks for both image and audio tasks.

## Project Structure

```bash
.
├── train.py            # Core logic: dataloading, model definitions, training & validation loops
├── __init__.py         # Global variable and parameter initialization
├── main.py             # Entry point of the pipeline; orchestrates training & evaluation
├── requirements.txt    # Python dependencies
```

### File Descriptions

- `main.py`: Drives the overall execution. Calls utility functions from `train.py`, sets up configurations, and runs training and evaluation.
- `train.py`: Implements dataset loading, custom CNNs, training and validation loops, and accuracy tracking.
- `__init__.py`: Initializes global variables, constants, and hyperparameters.
- `requirements.txt`: Lists all required packages to replicate the environment.

## Datasets

- **CIFAR-10**  
  - ```60,000``` ```32×32 color images``` across ```10 classes```.
  - Used for evaluating image classification performance.
  - [Link to dataset](https://www.cs.toronto.edu/~kriz/cifar.html)

- **SpeechCommands V0.02**  
  - Over ```100,000 one-second audio clips``` of ```35 spoken words```.
  - Used for evaluating audio classification robustness.
  - [Link to dataset](https://www.tensorflow.org/datasets/catalog/speech_commands)

## Technologies & Concepts

- **Languages**: ```Python```  
- **Frameworks**: ```PyTorch```, ```TensorFlow```  
- **Libraries**: ```Torchaudio```, ```Torchvision```, ```NumPy```, ```Matplotlib```  
- **Development**: ```Jupyter Notebooks```, ```Google Colab```  
- **Concepts**:  
  - ```Deep Learning```, ```CNNs```  
  - ```Cross-domain learning```  
  - ```Training pipelines``` and ```checkpointing```  
  - ```State-of-the-Art image/audio classification```  

## Setup Instructions

1. **Clone the Repository**
   ```bash
   git clone https://github.com/imtanmay46/Deep-Learning.git
   cd Deep-Learning
   ```

2. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Download Datasets**

   Although the ```train.py``` file handles everything, including the loading of the dataset, but if facing any issues:
   - Download ```CIFAR-10``` and ```SpeechCommands V0.02``` using the links above and do the necessary changes in ```train.py```.
  
5. Run the Training Pipeline
   ```bash
   python main.py
   ```
