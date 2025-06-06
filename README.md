# Image Classification and Model Distillation

This project was independently completed by **Yixuan Huang**, an undergraduate student from Wuhan University of Technology, as part of a major course project in Machine Learning. All code, analysis, and documentation were solely written and maintained by the author.

This project focuses on comparing two classical machine learning models for image classification tasks—**CNN** and **ResNet**—and introduces the concept of **model distillation**, where a stronger "teacher" model is used to guide the training of a simpler "student" model.

## Requirements

The project uses the following Python libraries:

- `scikit-learn`
- `medmnist`
- `torchvision`
- `matplotlib`
- `seaborn`

To install all dependencies, run:

```bash
pip install -r requirements.txt
```

## Training Results

### Accuracy Summary

| Model                     | Test Accuracy |
| ------------------------- | ------------- |
| SimpleCNN                 | 77.23%        |
| ResNet18                  | 81.91%        |
| Teacher CNN               | 98.81%        |
| Student (Distillation)    | 98.72%        |
| Student (No Distillation) | 98.64%        |

> 🔍 Knowledge Distillation improves student accuracy from **98.64% → 98.72%**, closely matching the teacher's performance.

### Training Loss Curves (Epochs = 5)

| Model                  | Final Training Loss |
| ---------------------- | ------------------- |
| SimpleCNN              | 0.2575              |
| ResNet18               | 0.1582              |
| Teacher CNN            | 0.0123              |
| Student (Distillation) | 0.1191              |
| Student (No Distill)   | 0.0193              |

## Classification Reports

### 🧩 SimpleCNN

- Test Accuracy: **77.23%**
- Struggled with digits like 2, 5, and 7 (lower F1-scores)
- Weighted average F1-score: **77.36%**

### 🧩 ResNet18

- Test Accuracy: **81.91%**
- Improved precision and recall on most classes
- Weighted average F1-score: **82.70%**

### 🧩 Distilled Student

- Test Accuracy: **98.72%**
- Very close to teacher model (98.81%)
- Great generalization in small model

## Summary

- **SimpleCNN** is a fast and lightweight model but has limited accuracy.
- **ResNet18** offers improved accuracy with deeper architecture.
- **Knowledge Distillation** enables a **compact model** to retain high accuracy with much fewer parameters.
- Distillation effectively **compresses** the teacher model without significant loss in performance.