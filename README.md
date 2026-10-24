# Handwritten Digit Classification



This project designs and optimizes an end-to-end machine learning pipeline to classify a sparse subset of the MNIST handwritten digit benchmark. Exploratory data analysis reveals an $81.0$% global pixel sparsity across the $784$-dimensional coordinate space, motivating dimensionality reduction, morphological edge extraction, and representation learning. Initial $5$-fold cross-validation benchmarking on raw pixel vectors identified the Support Vector Classifier with an RBF kernel ($\sim 96.0$%) and a Convolutional Neural Network ($97.5$%) as the leading classical and neural architectures, respectively. Integrating PCA ($50$ components) and deep Autoencoder latent embeddings ($d=64$) with the SVC compressed the feature space by over $90$% and reduced training times by approximately $75$% while improving cross-validation accuracy to $96.8$%. Extracting Histogram of Oriented Gradients (HOG) features enabled the RBF-SVC to achieve $98.1$% accuracy, matching the spatial capability of the CNN. On the held-out test partition, both HOG-SVC and CNN demonstrated generalization with identical test accuracies exceeding $98.2$%, significantly outperforming the $k$-NN baseline ($93.7$%) and exhibiting superior boundary robustness on structurally ambiguous digit pairs.



### Workflow



<img width="1462" height="802" alt="Image" src="https://github.com/user-attachments/assets/22913342-4941-4838-a720-c8d110e1af98" />



---



## Table of Contents

- [Overview & Problem Statement](#overview--problem-statement)

- [Dataset & Exploratory Analysis](#dataset--exploratory-analysis)

- [Pipeline Architecture](#pipeline-architecture)

  - [Preprocessing](#preprocessing)

  - [Feature Extraction & Dimensionality Reduction](#feature-extraction--dimensionality-reduction)

  - [Model Implementations](#model-implementations)

- [Experimental Setup & Hyperparameter Tuning](#experimental-setup--hyperparameter-tuning)

- [Benchmark Results](#benchmark-results)

  - [5-Fold Cross-Validation Leaderboard](#5-fold-cross-validation-leaderboard)

  - [Supervised vs. Semi-Supervised vs. Unsupervised Learning](#supervised-vs-semi-supervised-vs-unsupervised-learning)

  - [Final Hold-Out Test Evaluation](#final-hold-out-test-evaluation)

- [Repository Structure](#repository-structure)

- [Installation & Getting Started](#installation--getting-started)

- [Authors](#authors)



---



## Overview & Problem Statement



Classifying handwritten digits in high-dimensional coordinate space presents fundamental challenges:

1. **High Dimensionality & Pixel Sparsity:** Flattened $28 \times 28$ image matrices create a 784-dimensional feature space where over 81% of pixel values are strictly zero.

2. **Computational vs. Accuracy Trade-Offs:** Distance- and kernel-based algorithms (such as $k$-NN and kernel SVMs) suffer from quadratic-to-cubic complexity scaling when handling dense, raw pixel inputs.

3. **Representation Learning:** Evaluating how geometric edge descriptors (HOG), orthogonal linear projections (PCA), and non-linear deep bottlenecks (Autoencoders) mitigate redundancy while retaining class separability.

4. **Learning Paradigms:** Investigating performance under varying label availability regimes: fully supervised, semi-supervised (unsupervised representation pretraining followed by fine-tuning), and unsupervised clustering ($k$-Means).



---



## Dataset & Exploratory Analysis



The dataset is a curated subset of the MNIST benchmark provided in tabular CSV format:

- **Total Valid Observations:** $N = 8{,}042$ samples across 10 balanced digit classes ($0$ through $9$).

- **Data Integrity:** A single corrupted trailing line at index 8,042 containing 409 missing entries was audited and dropped.

- **Pixel Intensity Range:** $[0.0, 255.0]$ integer grayscale values.

- **Spatial Sparsity:** $81.0\%$ global sparsity (mean of 149 non-zero pixels per image; 120 boundary pixels are identically zero across all instances).

- **Class Balance:** Approximately uniform distribution across all 10 digit classes (\~804 instances/class).



---



## Pipeline Architecture



### Preprocessing

- **Deterministic Scaling:** Scaled to the unit interval $[0.0, 1.0]$ using a global constant factor ($x / 255.0$). This eliminates distribution-dependent leakage across cross-validation folds.

- **Stratified Partitioning:** A fixed random seed (`SEED = 42`) splits data into an $80\%$ development set ($N\_{\text{train}} = 6{,}433$) and a strictly isolated $20\%$ hold-out test set ($N\_{\text{test}} = 1{,}609$).

- **Disregarded Steps:** Binarization, denoising blur filters, and synthetic oversampling/class-weighting were deliberately omitted to prevent loss of sub-pixel edge transitions.



### Feature Extraction & Dimensionality Reduction

All feature extractors are packaged as `scikit-learn` pipeline transformers to guarantee zero training fold leakage:

- **PCA (`PCA`):** Truncates input space down to $n\_{\text{comp}} = 50$ components (which captures $\sim 85\%$ total variance; 84 components capture 90%, and 147 capture 95%).

- **HOG (`HOG`):** Reshapes flat vectors into native $28 \times 28$ arrays and extracts block-normalized gradients using $4 \times 4$ pixel cells, 9 orientation bins, and $2 \times 2$ cells per block ($L_2\text{-Hys}$ normalization).

- **Deep Autoencoder (`AEFeatures`):** Fully connected symmetric bottleneck ($\text{Encoder}: 784 \rightarrow 512 \rightarrow 256 \rightarrow 64$; $\text{Decoder}: 64 \rightarrow 256 \rightarrow 512 \rightarrow 784$) trained on unlabeled images via MSE reconstruction loss. Downstream models ingest the $64$-dimensional latent activations $Z = f\_\theta(X) \in \mathbb{R}^{64}$.



### Model Implementations

- **Classical Estimators (`scikit-learn`):** Logistic Regression (L-BFGS), $k$-Nearest Neighbors ($k$-NN), Support Vector Classifier with RBF kernel (`gamma='scale'`), and Random Forest (200 trees).

- **Custom Deep Networks (`PyTorch`):** Inherit from `BaseNN` with mini-batch SGD, Adam optimization, batched inference, and early stopping on internal validation folds:

  - **MLP:** $784 \rightarrow 512 \rightarrow 256 \rightarrow 128 \rightarrow 10$ with ReLU and Dropout ($p=0.3$).

  - **CNN:** Two sequential convolutional blocks ($\text{Conv2D}(1, 16, 3) \rightarrow \text{ReLU} \rightarrow \text{MaxPool2D}(2)$ and $\text{Conv2D}(16, 32, 3) \rightarrow \text{ReLU} \rightarrow \text{MaxPool2D}(2)$) followed by a dense classification head ($\text{Dropout}(0.3) \rightarrow \text{Linear}(1568, 128) \rightarrow \text{ReLU} \rightarrow \text{Dropout}(0.3) \rightarrow \text{Linear}(128, 10)$).



---



## Experimental Setup & Hyperparameter Tuning



All tuning and model selections are executed using **paired, stratified 5-fold cross-validation** on the $80\%$ training partition:



| Parameter | Model / Feature | Search Space | Selected Value | Justification |
| :--- | :--- | :--- | :--- | :--- |
| **$k$** | $k$-NN | $\\{1, 3, 5, 7, 11, 15\\}$ | **$k = 3$** | Accuracy peaked at $0.9367 \pm 0.0065$; balances local boundaries against noise. |
| **$C$** | SVC (RBF) | $\\{0.5, 1, 2, 5, 10, 20, 50\\}$ | **$C = 5.0$** | Plateaus at $0.9602 \pm 0.0054$; optimal margin without support vector inflation. |
| **$n\_{\text{comp}}$** | PCA + SVC | $\\{10, 20, 30, 50, 100, 150\\}$ | **$n\_{\text{comp}} = 50$** | Attains $0.9681 \pm 0.0041$; cuts SVM fitting time by $75\%$ while filtering pixel noise. |



---



## Benchmark Results



### 5-Fold Cross-Validation Leaderboard



| Model / Pipeline | CV Accuracy | Macro-$F_1$ | Fit Time (s) |
| :--- | :---: | :---: | :---: |
| **HOG + SVM (tuned, $C=5$)** | **0.9810 ± 0.0029** | **0.9810 ± 0.0029** | 7.9 |
| HOG + SVM (default, $C=1$) | 0.9792 ± 0.0034 | 0.9792 ± 0.0034 | 6.6 |
| **CNN** | **0.9754 ± 0.0060** | **0.9754 ± 0.0059** | 5.0 |
| PCA-50 + SVM (tuned, $C=5$) | 0.9681 ± 0.0041 | 0.9679 ± 0.0041 | 0.7 |
| AE codes + SVM (tuned, $C=5$) | 0.9681 ± 0.0040 | 0.9680 ± 0.0040 | 0.5 |
| AE codes + SVM (default) | 0.9649 ± 0.0024 | 0.9647 ± 0.0024 | 0.4 |
| PCA-50 + SVM (default) | 0.9635 ± 0.0045 | 0.9633 ± 0.0045 | 0.4 |
| SVM (RBF, tuned $C=5$) | 0.9602 ± 0.0054 | 0.9599 ± 0.0053 | 1.8 |
| SVM (RBF, default $C=1$) | 0.9552 ± 0.0018 | 0.9549 ± 0.0017 | 1.7 |
| AE codes + $k$-NN (tuned, $k=3$) | 0.9545 ± 0.0057 | 0.9542 ± 0.0055 | 0.1 |
| HOG + $k$-NN | 0.9510 ± 0.0065 | 0.9512 ± 0.0064 | 2.5 |
| PCA-50 + $k$-NN (tuned, $k=3$) | 0.9476 ± 0.0028 | 0.9474 ± 0.0027 | 0.1 |
| MLP | 0.9462 ± 0.0033 | 0.9458 ± 0.0034 | 4.6 |
| Random Forest | 0.9447 ± 0.0030 | 0.9442 ± 0.0029 | 1.1 |
| $k$-NN (tuned, $k=3$) | 0.9367 ± 0.0065 | 0.9368 ± 0.0063 | 0.0 |
| Logistic Regression | 0.8949 ± 0.0039 | 0.8936 ± 0.0036 | 1.2 |



### Supervised vs. Semi-Supervised vs. Unsupervised Learning



- **Unsupervised Clustering ($k$-Means, $k=10$):** Clusters mapped post-hoc by majority class label. Achieved test accuracies of **$58.67\%$** on raw pixels ($\text{NMI} = 0.4913$, $\text{ARI} = 0.3594$), **$58.30\%$** on PCA-50, and **$53.14\%$** on Autoencoder latent codes. Confirms that pure unsupervised metric clustering struggles with topological digit overlap.

- **Semi-Supervised Pretraining (Label Availability Sweep):**

  - Evaluated across label fractions: $2\%$ ($128$ labels), $5\%$ ($321$), $10\%$ ($643$), $25\%$ ($1608$), and $100\%$ ($6433$).

  - Autoencoder pretraining showed consistent gains only in the low-label regime ($\sim 10\%$ labels), achieving an accuracy gain of $+1.06\% \pm 2.00\%$ over training from scratch. At 100% label availability, the fully supervised SVM and CNN dominated.



### Final Hold-Out Test Evaluation



The top candidates were refitted once on the entire $80\%$ training split ($N\_{\text{train}} = 6{,}433$) and evaluated on the unseen $20\%$ holdout test partition ($N\_{\text{test}} = 1{,}609$):



| Model | Test Accuracy | Macro-$F_1$ | Fit Time (s) | Parameters ($\vert{}\Theta\vert{}$) | Key Trade-Off |
| :--- | :---: | :---: | :---: | :---: | :--- |
| **CNN** | **0.9832** | **0.9832** | **0.017** | **206,922** | **Highest accuracy & fastest inference latency.** |
| **HOG + SVM (tuned)** | **0.9820** | **0.9819** | 4.402 | 3,629,250 | Matches CNN accuracy; explicit geometric explainability. |
| $k$-NN (tuned, $k=3$) | 0.9323 | 0.9319 | 0.185 | 5,043,472 | Baseline; queries scale linearly with stored training matrix. |



**Error Analysis:** Confusion matrix diagnostics show that the $(4 \leftrightarrow 9)$ pair represents the primary remaining source of misclassification across all models due to shared vertical stroke geometry, followed by minor ambiguities in $(3 \leftrightarrow 5)$ and $(7 \leftrightarrow 1)$.



---



## Repository Structure



```text
├── data/
│   └── data.csv              # MNIST subset (8,042 rows x 785 columns)
├── main.ipynb                # End-to-end execution notebook (Sections 1-5)
├── README.md                 # Project documentation and summary
└── requirements.txt          # Python environment dependencies
```

## Installation & Getting Started

### 1. Prerequisites

Ensure you have Python 3.9 or later installed, along with `pip` and virtual environment tools.

### 2. Clone the Repository

```bash
git clone https://github.com/BRohit31/Handwritten-Image-Classification.git
cd Handwritten-Image-Classification
```

### 3. Create a Virtual Environment and Install Dependencies

```bash
python3 -m venv venv

# macOS / Linux
source venv/bin/activate

# Windows (PowerShell)
.\venv\Scripts\Activate.ps1

# Windows (Command Prompt)
venv\Scripts\activate.bat

pip install --upgrade pip
pip install -r requirements.txt
```

Key packages used include `torch`, `torchvision`, `scikit-learn`, `scikit-image`, `numpy`, `pandas`, and `matplotlib`.

### 4. Run the Experiments

Launch Jupyter Notebook and open the end-to-end pipeline notebook:

```bash
jupyter notebook main.ipynb
```

Alternatively, launch JupyterLab with `jupyter lab` and open `main.ipynb`. Run the notebook cells in order.

---

## Authors

- **Rohit Bhar**
- **Avik Trilok**
