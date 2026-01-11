# Open-Source Map Recognition AI

An AI-powered system that identifies countries from Google Street View images and video streams using deep learning models running on NVIDIA GPUs.

![License](https://img.shields.io/badge/license-MIT-blue.svg)
![Python](https://img.shields.io/badge/python-3.10+-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)

## Overview

This project develops an AI system capable of identifying countries from images sourced from Google Street View or other video streams. It combines image processing with state-of-the-art deep learning models, supporting both static images and live video streams.

**Research Question**: Can a SigLIP2-based model accurately recognize countries from video streams, and how does it perform compared to existing systems like GeoGuessr-55?

## Features

- **Multi-Source Input Support**
  - Upload images directly
  - Live screenshot capture
  - External URL loading
  - Random local test images
  
- **Dual Model Architecture**
  - **Model A**: Fine-tuned SigLIP checkpoint (custom trained)
  - **Model B**: GeoGuessr-55 pre-trained model
  - Side-by-side comparison of predictions
  
- **Real-Time Processing**
  - Live video stream analysis
  - Automatic periodic screenshot capture
  - Instant classification with confidence scores
  
- **Interactive Visualization**
  - Top-K predictions with probability scores
  - Side-by-side bar charts
  - Optional center-crop preprocessing

## Architecture

### System Pipeline

**Input Layer** → **Processing Layer** → **Output Layer**

#### Input Sources
- Google Street View images
- Live video streams (GeoGuessr, webcam)
- Uploaded image files (PNG, JPG, JPEG)
- Screenshot capture (local desktop)
- External URLs

#### Processing Pipeline
1. **Image Preprocessing**
   - Resize to 224×224 pixels
   - Normalize with model-specific parameters
   - Optional center-crop for screenshots

2. **Model Inference** (Dual architecture)
   - **Model A**: Fine-tuned SigLIP2 checkpoint
   - **Model B**: GeoGuessr-55 pre-trained model
   - PyTorch backend with CUDA acceleration

3. **Post-processing**
   - Softmax probability calculation
   - Top-K prediction extraction
   - Confidence score formatting

#### Output
- Country labels with confidence scores
- Side-by-side model comparison
- Interactive bar charts
- Real-time visualization

#### Hardware Acceleration
- **GPU**: NVIDIA CUDA-enabled devices (recommended)
- **CPU**: Automatic fallback for systems without GPU
- **Mixed Precision**: FP16 training and inference support

## Requirements

### Hardware
- NVIDIA GPU with CUDA support (recommended for training)
- Minimum 8GB RAM (16GB+ recommended for training)
- CPU fallback supported for inference

### Software
- Python 3.10 or higher
- CUDA Toolkit (for GPU acceleration)

### Python Dependencies
```
streamlit
torch
transformers
pillow
pyautogui
numpy
opencv-python
requests
datasets
torchvision
```

## Installation

### Quick Start (Testing Only)

If you just want to test the system with the pre-trained GeoGuessr-55 model:

```bash
# 1. Clone the repository
git clone https://github.com/yourusername/Open-Source-Map-Recognition-AI.git
cd Open-Source-Map-Recognition-AI

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run the dashboard
streamlit run streamlit_web_dashboard.py
```

**Note**: On first run, Model B (GeoGuessr-55) will be automatically downloaded from HuggingFace. Model A will show an error until you train your own checkpoint or disable it in the sidebar.

### Full Installation (Training + Testing)

For training your own model and using both models:

#### 1. Clone the Repository
```bash
git clone https://github.com/yourusername/Open-Source-Map-Recognition-AI.git
cd Open-Source-Map-Recognition-AI
```

#### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

#### 3. Download and Prepare Dataset

Download the [GeoGuessr Images Dataset from Kaggle](https://www.kaggle.com/datasets/ubitquitin/geolocation-geoguessr-images-50k) and organize it with this structure:

```
dataset/
├── Argentina/
│   ├── image1.jpg
│   ├── image2.jpg
│   └── ...
├── Australia/
│   ├── image1.jpg
│   └── ...
├── Austria/
└── ...
```

Each subdirectory represents a country class containing training images for that country.

#### 4. Train Your Model

```bash
python train.py
```

Training will:
- Load the dataset from `./dataset/`
- Fine-tune the SigLIP2 base model
- Save checkpoints to `./checkpoints/checkpoint-XXXXX/`
- Take approximately 2-5 hours on a modern GPU

#### 5. Update Dashboard Configuration

After training completes, update the checkpoint path in `streamlit_web_dashboard.py`:

```python
MODEL_A_PATH = "./checkpoints/checkpoint-XXXXX"  # Replace XXXXX with your checkpoint number
```

#### 6. Run the Dashboard

```bash
streamlit run streamlit_web_dashboard.py
```

The web interface will open at `http://localhost:8501`

### Testing the System

#### Option 1: Upload an Image
1. Select "Upload image" from the sidebar
2. Click "Upload an image" 
3. Choose a Street View image
4. View predictions from both models

#### Option 2: Use Image URL
1. Select "Example image URL" from the sidebar
2. Paste an image URL (e.g., from Google Street View)
3. Press Enter to classify

#### Option 3: Live Screenshot (Local Only)
1. Select "Live screenshot (local)" from the sidebar
2. Adjust the screenshot interval (default: 1 second)
3. Click "Start live"
4. Open Google Street View or GeoGuessr in another window
5. Watch real-time predictions

#### Option 4: Test with Local Images
1. Place test images in a `tests/` folder
2. Select "Random Local Image (auto)" if available
3. System will randomly select and classify images

### Troubleshooting Installation

**Missing checkpoint error for Model A**
- Either train your own model following step 4
- Or disable Model A in the dashboard sidebar and use only Model B

**CUDA not available**
- System will automatically fall back to CPU
- Training and inference will be slower but functional

**Package installation errors**
- Ensure you have Python 3.10 or higher: `python --version`
- Try upgrading pip: `pip install --upgrade pip`
- Install packages individually if needed

## Training

### Training Pipeline

The training process fine-tunes a SigLIP2 base model on geolocated images for country classification.

#### 1. Dataset Preparation

```python
from datasets import load_dataset

dataset = load_dataset("imagefolder", data_dir="dataset")
label_names = dataset["train"].features["label"].names
```

The dataset is automatically loaded from the `dataset/` directory using the `imagefolder` format, where each subdirectory name becomes a class label.

#### 2. Label Mapping

```python
id2label = {i: l for i, l in enumerate(label_names)}
label2id = {l: i for i, l in enumerate(label_names)}
num_classes = len(label_names)
```

Creates bidirectional mappings between numeric IDs and country names for model training and inference.

#### 3. Train/Validation Split

```python
dataset = dataset["train"].train_test_split(test_size=0.1)
```

- **Training set**: 90% of data
- **Validation set**: 10% of data

#### 4. Image Preprocessing

```python
from torchvision.transforms import Compose, Resize, ToTensor, Normalize

image_transforms = Compose([
    Resize((224, 224)),
    ToTensor(),
    Normalize(mean=processor.image_mean, std=processor.image_std),
])
```

All images are:
- Resized to 224×224 pixels
- Converted to tensors
- Normalized using SigLIP's mean and standard deviation

#### 5. Base Model

```python
model_name = "google/siglip2-base-patch16-224"
model = AutoModelForImageClassification.from_pretrained(
    model_name,
    num_labels=num_classes,
    label2id=label2id,
    id2label=id2label,
)
```

**Base Model**: Google's SigLIP2 (Sigmoid Loss for Language-Image Pre-training)
- Patch size: 16×16
- Input resolution: 224×224
- Pre-trained on large-scale image-text pairs

### Training Configuration

```python
TrainingArguments(
    output_dir="./checkpoints",
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    warmup_ratio=0.1,
    learning_rate=5e-5,
    num_train_epochs=10,
    save_strategy="epoch",
    eval_strategy="epoch",
    fp16=torch.cuda.is_available(),
    logging_steps=50,
    load_best_model_at_end=True,
)
```

#### Hyperparameters

| Parameter | Value | Description |
|-----------|-------|-------------|
| **Batch Size** | 16 | Per-device batch size for training and evaluation |
| **Learning Rate** | 5e-5 | Initial learning rate with warmup |
| **Warmup Ratio** | 0.1 | 10% of training steps for learning rate warmup |
| **Epochs** | 10 | Total number of training epochs |
| **Mixed Precision** | FP16 | Automatic mixed precision (if GPU available) |
| **Save Strategy** | epoch | Save checkpoint after each epoch |
| **Eval Strategy** | epoch | Evaluate model after each epoch |

#### Optimization Details

- **Optimizer**: AdamW (default in Hugging Face Trainer)
- **Learning Rate Schedule**: Linear warmup followed by linear decay
- **Best Model Selection**: Automatically loads the best checkpoint based on evaluation metrics
- **Checkpointing**: Model saved after each epoch to `./checkpoints/`

### Running Training

#### From Scratch
```bash
python train.py
```

#### Resume from Checkpoint
```python
trainer.train(resume_from_checkpoint=True)
```

This will automatically resume from the latest checkpoint in `./checkpoints/`.

### Training Monitoring

During training, the following metrics are logged every 50 steps:
- Training loss
- Validation loss
- Validation accuracy
- Learning rate

Monitor training progress in the console output or integrate with tools like TensorBoard or Weights & Biases.

### Expected Training Time

Approximate training time (varies based on hardware and dataset size):

| Hardware | Time per Epoch | Total Training Time |
|----------|----------------|---------------------|
| NVIDIA RTX 3090 | ~15-30 min | 2.5-5 hours |
| NVIDIA RTX 4090 | ~10-20 min | 1.5-3.5 hours |
| NVIDIA A100 | ~8-15 min | 1.5-2.5 hours |
| CPU (not recommended) | 6-10 hours | 60-100 hours |

*Based on ~50K images across 55 classes*

### Training Output

After training completes:
- Best model checkpoint saved in `./checkpoints/checkpoint-XXXXX/`
- Training logs in `./checkpoints/runs/`
- Model configuration and tokenizer files included

## Usage

### Start the Dashboard
```bash
streamlit run streamlit_web_dashboard.py
```

The web interface will open at `http://localhost:8501`

### Input Modes

#### 1. Upload Image
- Click "Upload an image"
- Select a PNG, JPG, or JPEG file
- View predictions instantly

#### 2. Live Screenshot (Local)
- Select interval (0.5-5 seconds)
- Click "Start live"
- System captures and classifies screenshots automatically
- Press "Stop live" or Ctrl+C to end

#### 3. Example Image URL
- Paste any image URL (http/https)
- Press Enter to load and classify

#### 4. Random Local Image
- Automatically selects from local dataset
- Useful for batch testing

### Model Controls

Use the sidebar to:
- Select input mode
- Enable/disable individual models
- Adjust screenshot interval
- Toggle center-crop preprocessing

## Supported Countries

Model B (GeoGuessr-55) classifies 55 countries:

<details>
<summary>View full list</summary>

Argentina, Australia, Austria, Bangladesh, Belgium, Bolivia, Botswana, Brazil, Bulgaria, Cambodia, Canada, Chile, Colombia, Croatia, Czechia, Denmark, Finland, France, Germany, Ghana, Greece, Hungary, India, Indonesia, Ireland, Israel, Italy, Japan, Kenya, Latvia, Lithuania, Malaysia, Mexico, Netherlands, New Zealand, Nigeria, Norway, Peru, Philippines, Poland, Portugal, Romania, Russia, Singapore, Slovakia, South Africa, South Korea, Spain, Sweden, Switzerland, Taiwan, Thailand, Turkey, Ukraine, United Kingdom

</details>

## Research Context

### Objectives
- Compare performance of SigLIP2-based models against GeoGuessr-55
- Evaluate platform independence and integration efficiency
- Benchmark against human performance
- Enable real-time country recognition from video streams

### Key Questions
1. How accurate is the fine-tuned SigLIP model compared to existing solutions?
2. Can the system operate effectively across different platforms?
3. What is the inference speed for real-time applications?
4. How does AI performance compare to human GeoGuessr players?

## Model Details

### Model A: Fine-tuned SigLIP2
- **Base**: `google/siglip2-base-patch16-224`
- **Architecture**: Vision Transformer (ViT) with patch-based processing
- **Training**: Fine-tuned on geolocated Street View images
- **Input Size**: 224×224 pixels
- **Output**: Country classification with confidence scores
- **Training Dataset**: 50K images across multiple countries
- **Training Time**: ~2-5 hours on modern GPU

### Model B: GeoGuessr-55
- **Source**: `prithivMLmods/GeoGuessr-55`
- **Pre-trained**: Ready for inference
- **Coverage**: 55 countries worldwide
- **Architecture**: SigLIP-based classification

## Configuration

### Dashboard Configuration

Key parameters in `streamlit_web_dashboard.py`:

```python
BASE_MODEL = "google/siglip2-base-patch16-224"
MODEL_A_PATH = "./checkpoints/checkpoint-28130"
MODEL_B_NAME = "prithivMLmods/GeoGuessr-55"
```

### Training Configuration

Key parameters in training script:

```python
model_name = "google/siglip2-base-patch16-224"
per_device_train_batch_size = 16
learning_rate = 5e-5
num_train_epochs = 10
```

Adjust these parameters based on your hardware capabilities and dataset size.

## Dataset Sources

- [GeoGuessr Images Dataset (Kaggle)](https://www.kaggle.com/datasets/ubitquitin/geolocation-geoguessr-images-50k) - 50K geolocated images
- Custom Street View extractions
- Organized by country for supervised learning


## Project Structure

```
Open-Source-Map-Recognition-AI/
├── streamlit_web_dashboard.py    # Main dashboard application
├── train.py                      # Training script
├── requirements.txt              # Python dependencies
├── README.md                     # This file
├── dataset/                      # Training dataset
│   ├── Argentina/
│   ├── Australia/
│   └── ...
├── checkpoints/                  # Model checkpoints
│   └── checkpoint-XXXXX/
└── tests/                        # Test images (optional)
```


## Acknowledgments

- [Google SigLIP2](https://huggingface.co/google/siglip2-base-patch16-224) for the base model
- [GeoGuessr-55](https://huggingface.co/prithivMLmods/GeoGuessr-55) for the comparison model
- [NVIDIA](https://developer.nvidia.com/cuda-toolkit) for CUDA toolkit and GPU acceleration
- [Hugging Face](https://huggingface.co/) for the Transformers library
- Kaggle community for the GeoGuessr dataset

## References

1. [Kaggle GeoGuessr Images Dataset](https://www.kaggle.com/datasets/ubitquitin/geolocation-geoguessr-images-50k)
2. [NVIDIA CUDA Toolkit Documentation](https://developer.nvidia.com/cuda-toolkit)
3. [NVIDIA AI Acceleration Documentation](https://developer.nvidia.com/accelerate-ai-applications/get-started)
4. [GPU Setup Guide for Deep Learning](https://github.com/entbappy/Setup-NVIDIA-GPU-for-Deep-Learning)
5. [GeoGuessr-55 Model (HuggingFace)](https://huggingface.co/prithivMLmods/GeoGuessr-55)
6. [Google SigLIP2 Model (HuggingFace)](https://huggingface.co/google/siglip2-base-patch16-224)

