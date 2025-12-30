# CSC173 Deep Vision Project Proposal
**Student:** Emmanuel Fitz C. Ciano, 2022-0154  
**Date:** December 11, 2025

## 1. Project Title 
Cancer Nuclei Analysis with EfficientNet

## 2. Problem Statement
Cancer diagnosis from histopathology images requires pathologists to manually locate and classify thousands of cells per slide, a process that is time-consuming (15-30 minutes per slide), subjective, and suffers from global pathologist shortages. Automated detection of cancer nuclei in histopathology images can significantly assist pathologists by providing fast, consistent binary segmentation (cancer vs background) to identify regions of interest for further analysis.

## 3. Objectives
- Develop a U-Net architecture with EfficientNet-B0 encoder for binary cancer nuclei segmentation
- Implement binary segmentation to distinguish cancer cells from background in histopathology images
- Achieve clinically relevant accuracy (>80% Dice coefficient for segmentation)
- Provide comprehensive evaluation metrics including Dice, IoU, and per-class accuracy
- Enable efficient inference suitable for clinical workflows

## 4. Dataset Plan
- **Primary Dataset:** PanNuke Dataset (histopathology images with multi-class masks)
  - 2,656 histopathology images (downsampled to 128x128)
  - 6 mask channels: Neoplastic cells, Inflammatory, Connective/Soft tissue cells, Dead Cells, Epithelial, Background
  - Binary segmentation mode: Combining channels 0-4 (cancer) vs channel 5 (background)
  - Train/validation split: 80/20 (2,124 training, 532 validation)
- **Domain:** Medical imaging, histopathology, cancer diagnosis

## 5. Technical Approach
- **Architecture:** U-Net with EfficientNet-B0 encoder for binary segmentation
- **Framework:** PyTorch with torchvision models
- **Key Techniques:** Transfer learning (ImageNet pre-trained EfficientNet-B0), binary segmentation, mixed precision training
- **Loss Function:** Combined Dice + BCE (Binary Cross-Entropy) with class weighting
- **Evaluation:** Medical imaging metrics (Dice coefficient, IoU, per-class accuracy)
- **Environment:** Local GPU (NVIDIA GeForce RTX 3050) with CUDA acceleration

## 6. Expected Challenges & Mitigations
- **Challenge:** Class imbalance (background vs cancer pixels ~5:1 ratio)
  - **Solution:** Class-weighted loss functions, weighted random sampling

- **Challenge:** Limited GPU memory (8GB VRAM)
  - **Solution:** Gradient accumulation, batch size of 4, mixed precision training (AMP)

- **Challenge:** Small image resolution (128x128) may lose fine details
  - **Solution:** Efficient downsampling with area interpolation, focus on binary segmentation task

- **Challenge:** Model interpretability for clinical trust
  - **Solution:** Attention visualization, prediction overlays, comprehensive metrics reporting