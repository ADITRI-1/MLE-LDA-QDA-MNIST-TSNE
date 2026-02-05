# MLE, LDA & QDA on MNIST with t-SNE Visualization

This repository implements Maximum Likelihood Estimation (MLE), Linear Discriminant Analysis (LDA), and Quadratic Discriminant Analysis (QDA) from scratch on the MNIST dataset for digit classification.  

The project follows a statistical learning approach assuming multivariate Gaussian class-conditional distributions and visualizes high-dimensional data using t-SNE.

---

## 📌 Objectives

- Estimate class-wise mean and covariance using MLE
- Implement LDA with shared covariance matrix
- Implement QDA with class-specific covariance matrices
- Evaluate classification accuracy
- Visualize data using t-SNE (scikit-learn)
- Analyze discriminant values of test samples

---

## 📂 Dataset

MNIST handwritten digit dataset (IDX format):

- Training images and labels  
- Test images and labels  

Only digits **0, 1, and 2** are used, with **100 samples per class** for balanced evaluation.

---

## ⚙️ Methodology

### 1. Data Preprocessing
- Load MNIST IDX files directly
- Sample balanced classes (0,1,2)
- Vectorize 28×28 images to 784-d vectors
- Normalize pixel values to [0,1]

### 2. Maximum Likelihood Estimation (MLE)

For each class \(c\):

**Mean:**

\[
\mu_c = \frac{1}{N_c} \sum x_i
\]

**Covariance:**

\[
\Sigma_c = \frac{1}{N_c} \sum (x_i - \mu_c)(x_i - \mu_c)^T
\]

---

### 3. Classification

#### LDA (Linear Discriminant Analysis)
- Uses a shared pooled covariance matrix
- Produces linear decision boundaries

#### QDA (Quadratic Discriminant Analysis)
- Uses class-specific covariance matrices
- Produces nonlinear decision boundaries

---

### 4. Evaluation

- Overall accuracy for LDA and QDA
- Class-wise accuracy for QDA
- Discriminant values for a sample test point

---

### 5. Visualization

t-SNE from scikit-learn is used to project high-dimensional data into 2D:

- Training set visualization
- Test set visualization  
- Color-coded by digit class

Official API:
https://scikit-learn.org/stable/modules/generated/sklearn.manifold.TSNE.html

---

## 📊 Results

| Model | Accuracy |
|------|---------|
| LDA | Reported in execution |
| QDA | Reported in execution |

(QDA typically outperforms LDA due to flexible covariance modeling.)

---

## 📈 Sample Visualizations

t-SNE plots reveal clear clustering between digit classes 0, 1, and 2.

---

## 📦 Installation

```bash
pip install -r requirements.txt

