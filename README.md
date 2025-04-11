# Heart Disease Diagnosis Model

## Overview
This project implements a machine learning model to diagnose heart disease using the **Heart Disease Health Indicators Dataset** from BRFSS 2015. The objective is to predict whether an individual has heart disease based on various health-related features, evaluating the model with metrics such as accuracy, precision, and recall.

## Dataset
- **Source**: [Heart Disease Health Indicators Dataset (BRFSS 2015)](https://www.kaggle.com/datasets/alexteboul/heart-disease-health-indicators-dataset)
- **Description**: The dataset contains 22 columns, including the target variable `HeartDiseaseorAttack` (0 for no heart disease, 1 for heart disease) and 21 features such as `HighBP`, `HighChol`, `BMI`, `Smoker`, `Diabetes`, `Age`, and others.
- **Size**: Approximately 253,680 rows.

## Requirements
To run the code, you need the following Python libraries:
- `pandas`
- `scikit-learn`

You can install them using:
```bash
pip install pandas scikit-learn
```

## Project Structure
- **Notebook**: `diagnosis.ipynb` (contains the complete code for data loading, preprocessing, model training, and evaluation)
- **Dataset**: Expected to be placed in the `/kaggle/input/heart-disease-health-indicators-dataset/` directory (or update the path in the code to match your local setup).

## Code Workflow
1. **Import Libraries**:
   - Libraries for data handling (`pandas`), model building (`RandomForestClassifier`), preprocessing (`StandardScaler`), and evaluation (`accuracy_score`, `precision_score`, `recall_score`, `classification_report`) are imported.

2. **Load and Inspect Data**:
   - The dataset is loaded using `pandas.read_csv()`.
   - The first 5 rows are displayed to verify the data structure.

3. **Feature Selection**:
   - The target variable is `HeartDiseaseorAttack`.
   - All other columns are used as features (`X`), and the target column is separated as `y`.

4. **Data Preprocessing**:
   - Features are standardized using `StandardScaler` to ensure uniformity in scale.

5. **Train-Test Split**:
   - The data is split into 70% training and 30% testing sets using `train_test_split` with a random state of 42 for reproducibility.

6. **Model Training**:
   - A `RandomForestClassifier` with 100 estimators is initialized and trained on the training data.

7. **Predictions**:
   - The trained model predicts outcomes on the test set.

8. **Model Evaluation**:
   - Metrics calculated: **Accuracy**, **Precision**, **Recall**.
   - A detailed classification report is generated, including per-class precision, recall, and F1-score.

## Results
The model achieves the following performance on the test set (results may vary slightly due to randomness):
- **Accuracy**: ~0.90
- **Precision**: ~0.87 (weighted average)
- **Recall**: ~0.90 (weighted average)
- **Classification Report**:
  - Class 0 (No Heart Disease): High precision (~0.92) and recall (~0.98).
  - Class 1 (Heart Disease): Lower precision (~0.44) and recall (~0.12), indicating challenges in predicting the minority class.

## Usage
To run the notebook:
1. Ensure the dataset is available at the specified path or update the file path in the code.
2. Install the required libraries (see Requirements).
3. Open the notebook in a Jupyter environment (e.g., Jupyter Notebook, Kaggle, or Google Colab).
4. Execute the cells sequentially.

Example command to start Jupyter Notebook:
```bash
jupyter notebook diagnosis.ipynb
```

## Notes
- **Class Imbalance**: The dataset is imbalanced, with fewer instances of heart disease (Class 1). This affects the model's performance on the minority class, as seen in the lower precision and recall for Class 1.
- **Improvements**: Consider techniques like oversampling (e.g., SMOTE), undersampling, or class-weight adjustments to handle imbalance. Hyperparameter tuning for the RandomForestClassifier could also improve performance.
- **Scalability**: The RandomForestClassifier is computationally intensive for large datasets. For faster experimentation, you can reduce `n_estimators` or sample the dataset.

## License
This project is for educational purposes and uses a publicly available dataset. Ensure compliance with the dataset's license terms when using or distributing.

## Acknowledgments
- Dataset provided by the Behavioral Risk Factor Surveillance System (BRFSS) 2015.
- Built using `scikit-learn` and `pandas` for machine learning and data processing.

For questions or contributions, feel free to contact the project author or submit a pull request.
