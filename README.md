# 🧬 Skin Cancer Classification - ISIC 2024  
A Deep Learning & Machine Learning Based Study

An in-depth research project to detect malignant skin lesions using image data and clinical metadata from the ISIC 2024 dataset. Combines the power of convolutional neural networks with classic ML models to uncover critical insights and reduce misdiagnoses.

---

## 🚀 Live Highlights

- ✅ Image-based classification using EfficientNet-B0  
- 🧠 Metadata fusion using Random Forest, SVC, Logistic Regression & LGBM  
- 📊 Rich evaluation: Accuracy, F1-score, AUC-ROC, Confusion Matrix  
- 🧪 Trained on 2196 samples with class imbalance handling  

---

---

## 📊 Dataset Overview

- **Source**: ISIC 2024 Challenge Dataset  
- **Records**: 2196 samples (after sampling)  
- **Features**: Skin lesion images, metadata (age, gender, anatomical site)  
- **Target**: Benign (0) or Malignant (1)

### 🔢 Class Distribution (Post-Sampling)

| Class     | Count | Percentage |
|-----------|--------|------------|
| Benign    | 1803   | 82.10%     |
| Malignant | 393    | 17.89%     |

---

## 🔍 Model Performance Comparison

| Model                  | Accuracy | AUC    | F1 (Malignant) | TN  | FP  | FN  | TP  |
|------------------------|----------|--------|----------------|-----|-----|-----|-----|
| EfficientNet-B0        | 78.65%   | 0.761  | 0.4977         | 686 | 133 | 83  | 107 |
| Random Forest          | 88%      | 0.9317 | 0.57           | 808 | 11  | 110 | 80  |
| Logistic Regression    | 73%      | 0.8979 | 0.56           | 572 | 247 | 21  | 169 |
| Support Vector Machine | 86%      | 0.9232 | 0.68           | 721 | 98  | 40  | 150 |
| LGBM Classifier        | 90%      | 0.9094 | 0.68           | 792 | 27  | 77  | 113 |

**Note**: TN = True Negatives, FP = False Positives, FN = False Negatives, TP = True Positives

---

## 💡 Key Insights

- Metadata like age, gender, and service features improve model performance when fused.
- Support services like tech help and backup correlate with lower malignancy in predictions.
- Logistic Regression performs well on recall but poorly on specificity.
- LGBM and SVC provide the most balanced precision-recall tradeoff.

---

## 🧠 Recommendations

- Fine-tune deeper layers of EfficientNet rather than freezing all.
- Implement early or late fusion of image and metadata in deep neural networks.
- Explore transformer-based multimodal architectures.
- Apply hyperparameter optimization via Optuna or GridSearchCV.
- Investigate class imbalance via targeted augmentation or GAN-based synthetic images.

---

## 🛠️ Tech Stack

- **Language**: Python  
- **Libraries**: PyTorch, Torchvision, Scikit-learn, Pandas, NumPy, Seaborn, Matplotlib  
- **Modeling**: EfficientNet-B0, RandomForest, LogisticRegression, LGBM, SVC  

---

## 📦 Installation

1. **Clone the Repository**
```bash
git clone https://github.com/your-username/skin-cancer-isic2024.git
cd skin-cancer-isic2024
```

## 🙋‍♂️ Author
Nirmit Gavate
📧 Email: gavatenirmit@gmail.com

🔗 GitHub: [@your-github](https://github.com/NirmitGavate)

🔗 LinkedIn: [LinkedIn Profile](https://www.linkedin.com/in/nirmit-gavate-4210262b1/)

📜 License
This project is licensed under the MIT License.
Feel free to use, share, and modify it with proper attribution.

⭐ Show Your Support
If you found this project helpful, feel free to give it a ⭐ star on GitHub!


