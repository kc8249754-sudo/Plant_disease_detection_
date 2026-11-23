# Plant Disease Detection with Deep Learning (EfficientNetB0)

Classifies leaf images into healthy / diseased classes using transfer learning, with Grad-CAM explainability and a test on real-world field images.

> Replace every `XX` below with your own results before publishing.

## Problem
Early detection of plant leaf diseases helps reduce crop loss. This project builds an image classifier that predicts the crop and disease (or "healthy") from a single leaf photo.

## Dataset
- **PlantVillage** (Kaggle: `abdallahalidev/plantvillage-dataset`, colour images), 38 classes across 14 crops.
- Exact duplicate images removed to prevent train/test leakage.
- Stratified split: 70% train / 15% validation / 15% test (split done before augmentation).

The dataset is not included in this repository. The notebook downloads it automatically with `kagglehub`.

## Approach
| Model | Description |
|---|---|
| **Baseline CNN** | 4 x (Conv, BatchNorm, ReLU, MaxPool), then GlobalAveragePooling, Dropout, Softmax. Trained from scratch. |
| **EfficientNetB0** | ImageNet-pretrained backbone with a new head (GlobalAveragePooling, Dropout 0.3, Dense softmax). |

**Training decisions**
- Two-phase transfer learning: frozen backbone (LR 1e-3), then fine-tuning the top 30 layers (LR 1e-5), BatchNorm layers kept frozen.
- Data augmentation (flip, rotation, zoom, brightness, contrast) applied during training only.
- Class weights to handle class imbalance.
- Adam optimizer, EarlyStopping, ReduceLROnPlateau and ModelCheckpoint on validation loss.
- Test set evaluated once, after all decisions were final.

## Results
| Model | Test accuracy | Macro F1 | Weighted F1 |
|---|---|---|---|
| Baseline CNN | XX% | XX | XX |
| EfficientNetB0 (fine-tuned) | XX% | XX | XX |

**Field-image test** (XX real-world leaf photos, not from PlantVillage): accuracy **XX%**, compared with XX% on the lab-style test split.

### Figures
![Learning curves](images/learning_curves.png)
![Confusion matrix](images/confusion_matrix.png)
![Grad-CAM](images/gradcam.png)

## Explainability
Grad-CAM heatmaps show which image regions drive each prediction. They are used to check whether the model focuses on the lesion or on the background.

## Limitations
- **Domain gap:** PlantVillage images are lab-style with plain backgrounds, so accuracy drops on field photos.
- **Leaf images only:** the model does not work on fruits, vegetables or whole plants, and only covers the 14 crops in the dataset.
- **Calibration:** softmax confidence can remain high on wrong predictions, including leaves from unsupported plants.
- **Class imbalance:** mitigated with class weights, but rare classes still have noisier metrics.
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
