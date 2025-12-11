# CSC172 Computer Vision Project Proposal
**Student:** Emmanuel Fitz C. Ciano, 2022-0154  
**Date:** December 11, 2025

## 1. Project Title 
Cancer Nuclei Analysis with Vision Transformers

## 2. Problem Statement
Cancer diagnosis from histopathology images requires pathologists to manually locate and classify thousands of cells per slide, a process that is time-consuming (15-30 minutes per slide), subjective, and suffers from global pathologist shortages. Current AI solutions treat cell segmentation (finding cells) and classification (identifying cancer types) as separate tasks, losing valuable contextual information. There is a need for a unified, interpretable AI system that can perform both tasks simultaneously while providing explanations for its decisions.

## 3. Objectives
- Develop a unified Vision Transformer model for simultaneous cancer cell instance segmentation and classification
- Implement attention visualization to explain model decisions to pathologists
- Achieve clinically relevant accuracy (>80% mAP for segmentation, >85% F1 for classification)
- Compare transformer performance against traditional CNN baselines
- Provide quantitative analysis of cancer cell morphology patterns

## 4. Dataset Plan
- **Primary Dataset:** Cancer Instance Segmentation and Classification Dataset (Kaggle)
  - 180 high-resolution histopathology images
  - YOLO format annotations with 5 cancer types
  - Contains bounding boxes and class labels for ~15,000 cancer nuclei
- **Alternative Dataset:** Cell Instance Segmentation Dataset (Kaggle) if needed
- **Domain:** Medical imaging, histopathology, cancer diagnosis

## 5. Technical Approach
- **Architecture:** Vision Transformer (ViT) with multi-task heads for segmentation and classification
- **Framework:** PyTorch with Hugging Face Transformers library
- **Key Techniques:** Multi-task learning, attention visualization, transfer learning
- **Evaluation:** Medical imaging metrics (mAP, Dice coefficient, F1-score) with interpretability analysis
- **Environment:** Google Colab Pro with GPU acceleration

## 6. Expected Challenges & Mitigations
- **Challenge:** Small dataset size (180 images)
  - **Solution:** Heavy data augmentation, transfer learning from ImageNet

- **Challenge:** Class imbalance in cancer types
  - **Solution:** Weighted loss functions, focal loss, strategic sampling

- **Challenge:** Computational requirements for transformers
  - **Solution:** Gradient checkpointing, mixed precision training, cloud GPUs

- **Challenge:** Model interpretability for clinical trust
  - **Solution:** Attention visualization tools, Grad-CAM, attention rollout