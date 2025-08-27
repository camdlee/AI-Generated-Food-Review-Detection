# AI-Generated Food Review Detection: Multimodal Classification

## Project Overview
This project implements a machine learning pipeline to detect whether food reviews (paired with images) are AI-generated or authentic using supervised learning techniques. The system leverages both textual features from review content and visual features from accompanying images to create a robust multimodal classification approach.
- Goal: Build and evaluate models that can classify reviews (paired with images) as either authentic or AI-generated
- Data:
  - Training set: 12,086 rows
  - Test set: 4,030 rows
  - Validation set: 4,028 rows
  - Each dataset contains 25 features and is balanced (~50% authentic, ~50% AI-generated)

# Technologies Used
- Python (NumPy, Pandas, Matplotlib, Scikit-learn, seaborn)
- Jupyter Notebook for development and visualization

## Key Features
- Data Preprocessing
    - Loads training and testing datasets from CSV files.
    - Processes and combines text features with image-based features.
    - Utility functions for loading image features and preparing data.

- Implemented Functions
    - load_image_features: Loads and structures image-based features.
    - adaboost_train_predict: Implements AdaBoost algorithm from scratch.
    - evaluate_adaboost_cv / evaluate_adaboost_unstrat_cv: Runs cross-validation for AdaBoost.
    - evaluate_random_forest_cv / evaluate_random_forest_unstrat_cv: Evaluates Random Forest with cross-validation.
    - evaluate_svm_cv / evaluate_svm_unstrat_cv: Evaluates SVM with cross-validation.
    - summarize: Aggregates and reports evaluation results.

- Models & Techniques
    - Decision Tree (weak learner): Simple classifier with limited depth/nodes.
    - AdaBoost: Boosting ensemble combining weak learners.
    - Random Forest: Bagging-based ensemble of decision trees.
    - SVM: Kernel-based classification for complex decision boundaries.

- Evaluation Metrics
    - Accuracy: Overall correctness of predictions.
    - Precision, Recall, F1-score: Detailed class-level performance metrics.
    - Cross-validation: Both stratified and unstratified to test robustness.
  
## Key Findings
- AdaBoost Decision Tree achieved the highest performance (98.91% accuracy) by learning from previous learners' mistakes and effectively capturing nuanced differences between authentic and AI-generated content
- Random Forest performed well (97.29% accuracy) with good generalization due to ensemble averaging, but lacked the sequential error correction of boosting methods
- SVM showed the lowest performance (81.15% accuracy), likely due to the non-linear relationship between textual and image features that couldn't be effectively captured by hyperplane separation 
- Minimal difference between stratified and unstratified cross-validation (within 0.4%) due to balanced dataset (approximately 50% authentic, 50% AI-generated)
- Dataset balance makes accuracy the most suitable evaluation metric