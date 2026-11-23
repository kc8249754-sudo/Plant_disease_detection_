# Plant Disease Detection with Deep Learning (EfficientNetB0)

Classifies leaf images into 38 healthy / diseased classes using transfer learning, with Grad-CAM explainability and error analysis.

**Result:** EfficientNetB0 reaches **97.21% test accuracy** (macro F1 0.964) on 8,143 unseen test images, compared with 94.79% for a CNN trained from scratch.

## Problem
Early detection of plant leaf diseases helps reduce crop loss. This project builds an image classifier that predicts the crop and disease (or "healthy") from a single leaf photo.

## Dataset
- **PlantVillage** (Kaggle: `abdallahalidev/plantvillage-dataset`, colour images): 54,305 images, 38 classes, 14 crops.
- 21 exact duplicate images removed to prevent train/test leakage (54,284 remaining).
- Class imbalance ratio is 36.2 (largest class about 5,500 images, smallest about 150).
- Stratified split done **before** augmentation: 37,998 train / 8,143 validation / 8,143 test.

The dataset is not included in this repository. The notebook downloads it automatically with `kagglehub`.

## Approach
| Model | Description |
|---|---|
| **Baseline CNN** | 4 x (Conv, BatchNorm, ReLU, MaxPool), then GlobalAveragePooling, Dropout, Softmax. About 0.4M parameters, trained from scratch. |
| **EfficientNetB0** | ImageNet-pretrained backbone with a new head (GlobalAveragePooling, Dropout 0.3, Dense softmax). |

**Training decisions**
- Two-phase transfer learning: frozen backbone (LR 1e-3), then fine-tuning the top 30 layers (LR 1e-5), BatchNorm layers kept frozen for stability.
- Data augmentation (flip, rotation, zoom, brightness, contrast), applied during training only.
- Class weights to handle class imbalance.
- Adam optimizer, EarlyStopping, ReduceLROnPlateau and ModelCheckpoint on validation loss.
- Test set evaluated once, after all decisions were final.

## Results
| Model | Test accuracy | Macro F1 | Weighted F1 |
|---|---|---|---|
| Baseline CNN | 94.79% | 0.942 | 0.948 |
| EfficientNetB0 (fine-tuned) | **97.21%** | **0.964** | **0.972** |

The baseline's validation curve was unstable (large dips at several epochs), while fine-tuned EfficientNetB0 improved smoothly. Validation accuracy rose steadily during fine-tuning and reached about 97.7%.

### Figures
![Learning curves](images/learning_curves.png)
![Confusion matrix](images/confusion_matrix.png)
![Grad-CAM](images/gradcam.png)

## Error analysis
Most classes score F1 of 0.97 or higher. The weakest classes (EfficientNetB0):

| Class | F1 | Main confusion |
|---|---|---|
| Potato healthy | 0.759 | Only 23 test images; low precision (0.629) because a few other potato images are predicted as healthy |
| Tomato Early blight | 0.835 | Mistaken for Target Spot and Late blight |
| Corn Cercospora / Gray leaf spot | 0.863 | Mistaken for Northern Leaf Blight |
| Tomato Target Spot | 0.868 | Attracts errors from Early blight, Spider mites and Septoria |
| Tomato Spider mites | 0.902 | Mistaken for Target Spot |

Most errors are between diseases that look similar on the leaf, mainly among tomato classes and between two corn leaf diseases.

**Confidence:** mean softmax confidence is 0.982 on correct predictions and 0.683 on wrong ones, so low confidence is a useful warning sign. A small number of wrong predictions still have confidence above 0.95.

## Explainability
Grad-CAM heatmaps show which image regions drive each prediction. In most samples the model focuses on lesions (for example grape black rot and pepper bacterial spot). In one apple black rot sample, part of the attention fell on the leaf stem, which suggests the model can partly rely on non-lesion cues.

## Limitations
- **Domain gap:** PlantVillage images are lab-style with plain backgrounds, so accuracy is expected to drop on field photos.
- **Leaf images only:** the model does not work on fruits, vegetables or whole plants, and only covers the 14 crops in the dataset.
- **Calibration:** some wrong predictions are made with high confidence, including leaves from unsupported plants.
- **Class imbalance:** mitigated with class weights, but small classes (such as potato healthy) have noisy metrics.
- **Similar-looking diseases:** tomato diseases and the two corn leaf diseases are the main error source.
- Sensitive to lighting, blur and leaf orientation; one leaf per image; no severity estimation.

## Future work
- Train with field-condition images (for example PlantDoc) and stronger augmentation.
- Add confidence calibration (temperature scaling) and an "unknown / not a leaf" class.
- Test-time augmentation and a lightweight model for mobile deployment.

## How to run
1. Open `plant_disease_detection.ipynb` in Google Colab (Runtime, then Change runtime type, then GPU).
2. Run all cells from top to bottom. The dataset downloads automatically.
3. Optional: upload your own field photos to `/content/field_images/<class_name>/` and run the field-test section.

Or run locally:
```bash
pip install -r requirements.txt
jupyter notebook plant_disease_detection.ipynb
```

## Tech stack
Python, TensorFlow/Keras, scikit-learn, pandas, matplotlib, seaborn, Google Colab.

## Acknowledgements
- PlantVillage dataset (Hughes and Salathe, 2015).
- EfficientNet (Tan and Le, 2019), ImageNet pretrained weights from Keras Applications.
