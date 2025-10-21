# Fingerprint Spoofing Detection

This repository contains the complete project for the **Machine Learning and Pattern Recognition** course (A.Y. 2023/2024) at Politecnico di Torino.

The project's goal is the development and evaluation of a binary classification system to distinguish between **genuine** and **counterfeit** fingerprints.

**No high-level ML libraries (such as scikit-learn) were used.** All classifiers, pre-processing functions, and evaluation metrics were written manually, primarily using NumPy.

This approach allowed for a deep understanding of the internal mechanics of each model, including:
* Bayesian Classifiers (MVG, Naive Bayes, Tied Covariance)
* Logistic Regression (Linear and Quadratic)
* Support Vector Machines (SVM) (with Linear, Polynomial, and RBF kernels)
* Gaussian Mixture Models (GMM)
* Principal Component Analysis (PCA) and Linear Discriminant Analysis (LDA)
* Cost functions (DCF) and score calibration

## 📈 Project Pipeline

The project followed a structured development and validation pipeline:

1.  **Data Analysis and Visualization**: Exploratory analysis of the dataset (composed of 6 features) to understand the distributions of the "genuine" and "counterfeit" classes.
2.  **Dimensionality Reduction**: Implementation and testing of **PCA** and **LDA** to analyze class separability and reduce the feature space.
3.  **Modeling and Evaluation**: Implementation, training, and validation of multiple classifier families:
    * **Gaussian Models**: Comparison of Multi-variate Gaussian (MVG), Naive Bayes, and Tied Covariance.
    * **Logistic Regression**: Evaluation of linear and quadratic models (with feature expansion), analyzing the impact of regularization ($\lambda$).
    * **Support Vector Machines (SVM)**: Grid search for the best hyperparameters (C, $\gamma$) for linear, polynomial, and RBF kernels.
    * **Gaussian Mixture Models (GMM)**: Performance analysis by varying the number of components and covariance type (diagonal vs. full).
4.  **Evaluation Metric**: Performance analysis was conducted using the **Detection Cost Function (DCF)**, specifically comparing `minDCF` (discrimination) and `actDCF` (calibration) for different target applications (priors).
5.  **Selection, Calibration, and Fusion**:
    * Selection of the top three models: GMM, Quadratic Logistic Regression, and SVM (RBF).
    * Implementation of score calibration.
    * Evaluation of a final system based on **score-level fusion** of the best-performing models.

## Results and Final Model

Among all tested models, the most promising candidate identified during validation was the **Gaussian Mixture Model (GMM) with 8 components and diagonal covariance matrices**.

This model offered the best balance of discriminative power and calibration, achieving on the validation set:
* `minDCF`: 0.1463
* `actDCF`: 0.1809

On the final evaluation set, the 8-component diagonal GMM confirmed its excellent performance (achieving `minDCF` 0.203 and `actDCF` 0.221).
