# Cancer Nuclei Analysis with EfficientNet

**Project:** CSC173 Deep Vision Final Project  
**Student:** Emmanuel Fitz C. Ciano, 2022-0154  
**Date:** December 2025

## Abstract

This project addresses the critical challenge of automated cancer nuclei detection in histopathology images, a time-consuming and subjective task for pathologists. We developed a binary segmentation system using a U-Net architecture with EfficientNet-B0 encoder to distinguish cancer cells from background in histopathology images. The model was trained on 2,656 histopathology images from the PanNuke dataset, downsampled to 128×128 pixels, with 6 mask channels combined into binary segmentation (cancer vs background). Using transfer learning from ImageNet pre-trained weights, mixed precision training, and a combined Dice+BCE loss function, the model achieved a validation Dice coefficient of 0.6791 and IoU of 0.5825. While the target of 80% Dice was not fully met, the system demonstrates the feasibility of automated cancer nuclei detection and provides a foundation for further improvements. Key contributions include a lightweight architecture suitable for 8GB GPUs, comprehensive evaluation metrics, and attention visualization for clinical interpretability.

## Table of Contents

- [Introduction](#introduction)
- [Related Work](#related-work)
- [Methodology](#methodology)
- [Experiments & Results](#experiments--results)
- [Discussion](#discussion)
- [Ethical Considerations](#ethical-considerations)
- [Conclusion](#conclusion)
- [Installation](#installation)
- [References](#references)

## Introduction

### Problem Statement

Cancer diagnosis from histopathology images requires pathologists to manually locate and classify thousands of cells per slide, a process that is time-consuming (15-30 minutes per slide), subjective, and suffers from global pathologist shortages. Automated detection of cancer nuclei in histopathology images can significantly assist pathologists by providing fast, consistent binary segmentation to identify regions of interest for further analysis. This is particularly important in resource-limited settings where pathologist availability is constrained.

### Objectives

- **Objective 1:** Achieve >80% Dice coefficient for binary segmentation of cancer nuclei
- **Objective 2:** Develop a lightweight model suitable for deployment on standard GPUs (8GB VRAM)
- **Objective 3:** Provide interpretable results through attention visualization and comprehensive metrics

## Related Work

- **EfficientNet:** Tan & Le (2019) introduced EfficientNet, a family of models that achieve better accuracy with fewer parameters through compound scaling. EfficientNet-B0 serves as an efficient backbone for medical image segmentation tasks.

- **U-Net:** Ronneberger et al. (2015) proposed U-Net, a convolutional network architecture for biomedical image segmentation. The encoder-decoder structure with skip connections has become a standard for medical image segmentation.

- **Transfer Learning:** Pre-trained models on ImageNet have shown significant improvements in medical imaging tasks, allowing for better feature extraction with limited medical datasets.

- **Gap:** Our approach combines EfficientNet-B0 encoder with U-Net decoder specifically optimized for binary cancer nuclei segmentation on histopathology images, with focus on memory efficiency for standard GPUs.

## Methodology

### Dataset

- **Source:** PanNuke Dataset (histopathology images with multi-class segmentation masks)
- **Size:** 2,656 histopathology images
- **Resolution:** Downsampled from 256×256 to 128×128 pixels
- **Mask Channels:** 6 channels (Neoplastic cells, Inflammatory, Connective/Soft tissue cells, Dead Cells, Epithelial, Background)
- **Binary Mode:** Channels 0-4 combined as "cancer" vs channel 5 as "background"
- **Split:** 80/20 train/validation (2,124 training, 532 validation)
- **Preprocessing:** 
  - Image normalization to [0, 1] range
  - ImageNet normalization (mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
  - Data augmentation: Random horizontal/vertical flips
  - Mask downsampling with nearest-neighbor interpolation

### Architecture

**Model Diagram:**

```
Input Image (128×128×3)
    ↓
EfficientNet-B0 Encoder (ImageNet pre-trained)
    ├── Encoder Block 0: [B, 24, H/2, W/2]
    ├── Encoder Block 1: [B, 24, H/2, W/2]
    ├── Encoder Block 2: [B, 40, H/4, W/4]
    ├── Encoder Block 3: [B, 112, H/8, W/8]
    ├── Encoder Block 4: [B, 192, H/16, W/16]
    └── Encoder Block 5: [B, 1280, H/32, W/32]
    ↓
U-Net Decoder with Skip Connections
    ├── Decoder Block 4: [B, 192, H/16, W/16]
    ├── Decoder Block 3: [B, 112, H/8, W/8]
    ├── Decoder Block 2: [B, 40, H/4, W/4]
    ├── Decoder Block 1: [B, 24, H/2, W/2]
    └── Decoder Block 0: [B, 16, H, W]
    ↓
Final Convolution (1×1)
    ↓
Binary Segmentation Output [B, 1, 128, 128]
```

**Backbone:** EfficientNet-B0 (ImageNet pre-trained)  
**Head:** U-Net decoder with bilinear upsampling and skip connections  
**Total Parameters:** 19,480,839 (all trainable)

### Hyperparameters

| Parameter | Value |
|-----------|-------|
| Batch Size | 4 |
| Gradient Accumulation Steps | 8 |
| Effective Batch Size | 32 |
| Learning Rate | 1e-4 |
| Weight Decay | 1e-4 |
| Optimizer | AdamW |
| Scheduler | ReduceLROnPlateau (factor=0.5, patience=5) |
| Epochs | 20 |
| Loss Function | Dice + BCE (weighted) |
| Image Size | 128×128 |
| Mixed Precision | Yes (AMP) |

### Training Code Snippet

```python
model = EfficientNetSegmentation(
    img_size=128,
    in_channels=3,
    num_classes=5,
    dropout=0.2
).to(device)

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=1e-4,
    weight_decay=1e-4
)

criterion = BinarySegLoss(
    class_weights=class_weights_tensor,
    bce_weight=1.0,
    dice_weight=1.0
)

for epoch in range(num_epochs):
    train_loss = train_epoch(
        model, train_loader, criterion, optimizer, 
        device, scaler=scaler, use_amp=True, 
        accumulation_steps=8
    )
    val_metrics = evaluate_model(model, val_loader, criterion, device)
```

## Experiments & Results

### Metrics

| Model | Dice Coefficient | IoU | Background Accuracy | Cancer Accuracy | Inference Time (ms) |
|-------|------------------|-----|-------------------|-----------------|---------------------|
| Baseline (EfficientNet-B0) | - | - | - | - | - |
| Ours (Fine-tuned) | 0.6791 (best) | 0.5913 | 0.6398 | 0.6809 | ~15 |

**Final Validation Metrics:**
- Dice Coefficient: 0.6213
- IoU: 0.5825
- Background Accuracy: 0.6398
- Cancer Accuracy: 0.6809
- Overall Accuracy: 0.6603
- Validation Loss: 1.4251

### Training Curve

The model was trained for 20 epochs with early stopping patience of 5. Best validation Dice of 0.6791 was achieved at epoch 4. Training showed:
- Steady decrease in training loss
- Validation Dice improved from 0.5894 (epoch 1) to 0.6791 (epoch 4)
- Some overfitting observed in later epochs

### Demo

[\[Video: CSC173_Ciano_Final.mp4\]](https://drive.google.com/drive/folders/1ux-nKwPohlVyQKOwGwKc30GQY6C4P6Av?usp=drive_link)

## Discussion

### Strengths

- **Memory Efficient:** Model fits comfortably on 8GB GPU with gradient accumulation
- **Transfer Learning:** Leveraging ImageNet pre-trained EfficientNet-B0 provides strong feature extraction
- **Comprehensive Evaluation:** Multiple metrics (Dice, IoU, per-class accuracy) provide thorough assessment
- **Interpretability:** Attention visualization and prediction overlays aid clinical understanding
- **Robust Training:** Mixed precision training and gradient accumulation enable stable training

### Limitations

- **Target Not Met:** Dice coefficient of 0.6213 falls short of the 80% target
- **Low Resolution:** 128×128 images may lose fine-grained details important for accurate segmentation
- **Binary Segmentation:** Multi-class segmentation (5 cancer types) was simplified to binary, losing class-specific information
- **Limited Augmentation:** Only basic flips were used; more sophisticated augmentation could improve generalization
- **Short Training:** 20 epochs may be insufficient; longer training with learning rate scheduling could improve results

### Insights

- **Class Imbalance:** Background/cancer pixel ratio of ~5:1 required careful class weighting
- **Gradient Accumulation:** Effective batch size of 32 (via accumulation) was crucial for stable training on limited GPU memory
- **Mixed Precision:** AMP enabled faster training without significant accuracy loss
- **Early Stopping:** Best model was found early (epoch 4), suggesting potential for better hyperparameter tuning

## Ethical Considerations

- **Bias:** Dataset may be skewed toward certain cancer types or image acquisition conditions; rural or underrepresented populations may have different histopathology characteristics
- **Privacy:** Medical images used for training should be properly anonymized and comply with HIPAA/GDPR regulations
- **Misuse:** Automated cancer detection systems should be used as assistive tools, not replacements for pathologist judgment; false negatives could have serious consequences
- **Clinical Deployment:** Model performance (62% Dice) is below clinical acceptance thresholds; requires further validation before deployment

## Conclusion

This project successfully developed a binary segmentation system for cancer nuclei detection using EfficientNet-B0 and U-Net architecture. While the target Dice coefficient of 80% was not fully achieved (final: 62.13%, best: 67.91%), the system demonstrates the feasibility of automated cancer nuclei detection and provides a solid foundation for future improvements. Key achievements include a memory-efficient architecture suitable for standard GPUs, comprehensive evaluation metrics, and interpretable results through attention visualization.

**Future Directions:**
1. **Higher Resolution:** Train on 256×256 or 512×512 images to capture finer details
2. **Multi-class Segmentation:** Extend to 5-class segmentation to preserve cancer type information
3. **Advanced Augmentation:** Implement elastic transforms, color jitter, and mixup techniques
4. **Longer Training:** Extend training to 50-100 epochs with careful learning rate scheduling
5. **Architecture Improvements:** Experiment with DeepLabV3+, FPN, or other segmentation architectures
6. **Clinical Validation:** Evaluate on external test sets and collaborate with pathologists for clinical validation

## Installation

### Clone Repository

```bash
git clone https://github.com/yourusername/CSC173-DeepCV-Ciano
cd CSC173-DeepCV-Ciano
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

**requirements.txt:**
```
torch>=2.0
torchvision>=0.15.0
numpy>=1.24.0
opencv-python>=4.8.0
scikit-image>=0.21.0
scikit-learn>=1.3.0
matplotlib>=3.7.0
seaborn>=0.12.0
tqdm>=4.65.0
Pillow>=10.0.0
```

### Download Weights

Pre-trained model weights can be downloaded from the `models/` directory or by running:

```bash
# Model weights will be saved during training as 'best_model.pth'
# Load with: torch.load('best_model.pth', map_location=device)
```

### Dataset Setup

1. Download the PanNuke dataset
2. Organize into the following structure:
```
Dataset/
├── Images/
│   ├── images.npy
│   └── types.npy
└── Masks/
    └── masks.npy
```

3. Run the notebook `cancer_nuclei_analysis.ipynb` to preprocess and train

## References

[1] Tan, M., & Le, Q. V. (2019). EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks. *ICML*.

[2] Ronneberger, O., Fischer, P., & Brox, T. (2015). U-Net: Convolutional Networks for Biomedical Image Segmentation. *MICCAI*.

[3] Gamper, J., et al. (2020). PanNuke Dataset Extension, Insights and Baselines. *arXiv preprint arXiv:2003.10778*.

[4] Deng, J., et al. (2009). ImageNet: A large-scale hierarchical image database. *CVPR*.

[5] Paszke, A., et al. (2019). PyTorch: An Imperative Style, High-Performance Deep Learning Library. *NeurIPS*.

---

**GitHub Pages:** [View this project site](https://yourusername.github.io/CSC173-DeepCV-Ciano/)
