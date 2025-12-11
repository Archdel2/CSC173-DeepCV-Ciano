# Cancer Nuclei Analysis with Vision Transformers

## Project Overview
This project develops an AI system for automated cancer diagnosis by analyzing histopathology images. Using Vision Transformers, we perform **instance segmentation** (locating individual cancer cells) and **multi-class classification** (identifying cancer types) simultaneously in a unified model. The system provides interpretable results through attention visualization, bridging the gap between AI capabilities and clinical trust.

## Dataset Choice
**Selected:** Cancer Instance Segmentation and Classification Dataset from Kaggle

**Why this dataset:**
1. **Medical Relevance:** Direct application to cancer diagnosis - high impact potential
2. **Multi-task Ready:** Contains both segmentation masks (bounding boxes) and classification labels
3. **Research Novelty:** Not overused in competitions - opportunity for original contributions
4. **Transformer-Friendly:** Moderate size (180 images) suitable for experimentation
5. **Well-Structured:** Clean YOLO format annotations with 5 distinct cancer cell types

**Dataset Statistics:**
- 180 high-resolution histopathology images
- ~15,000 annotated cancer nuclei
- 5 cancer types: Healthy, Atypical, Pleomorphic, Giant, Epithelioid
- YOLO format annotations (bounding boxes + class labels)

## Architecture Sketch

### Core Design Principles:
1. **Unified Architecture:** Single transformer model handling both segmentation and classification
2. **Attention-Based:** Leverages transformer's self-attention for global context understanding
3. **Interpretable:** Built-in attention visualization for clinical explainability
4. **Multi-Scale Processing:** Handles varying cell sizes within histopathology images

Input Image (512×512 RGB)
    ↓
Patch Embedding + Position Encoding
    ↓
Vision Transformer Encoder
    (Multi-head self-attention layers)
    ↓
Multi-Task Feature Extraction
    ├── Segmentation Features → Instance Masks
    ├── Classification Features → Cancer Type Predictions
    └── Attention Features → Visualization Maps
    ↓
Outputs:
- Binary masks for each cancer cell
- Cancer type classification per cell (5 classes)
- Attention heatmaps showing model focus areas
