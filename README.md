# Naive Bayes Text Classification

A from-scratch implementation of standard Naive Bayes text classifiers and extensions. More precisely, this project explores feature selection, term-frequency transformations, inverse document frequency weighting, and Complement Naive Bayes.

Here, we investigate how these techniques affect the classification accuracy of a model trained on a collection of text abstracts. The project was part of a Kaggle competition hosted by the University of Auckland.

## Overview

Naive Bayes is a probabilistic classification algorithm based on Bayes' theorem. Despite its strong simplifying assumption that features are conditionally independent given the class, it is widely used for text classification due to its simplicity and efficiency.

This project implements a multinomial Naive Bayes classifier from scratch in Python, then explores several extensions intended to improve its performance on text data.

The main objectives were to:

- Implement a Naive Bayes classifier without relying on an existing machine learning library.
- Investigate the effect of vocabulary size and stop-word removal.
- Explore alternative term-frequency and document-frequency weightings.
- Implement Complement Naive Bayes to account for class imbalance.
- Compare the performance of the different approaches using stratified cross-validation.

## A Brief Introduction to Naive Bayes Classification

The goal of this project is to classify text documents (in this case, abstracts) into one of four categories. Each abstract is taken from a research paper concerning proteins and belongs to one of four categories: Archaea, Bacteria, Eukaryota, or Viruses. 

Given an unseen abstract, the classifier estimates the probability that it pertains to a particular protein category and assigns it to the most likely one.

Naive Bayes uses Bayes' theorem to calculate these probabilities. In the context of text classification, this involves combining:

- **Prior probabilities:** How frequently each category occurs in the training data.
- **Word likelihoods:** How likely each word is to occur in a document belonging to a particular category.

For a document containing words \(w_1, w_2, \ldots, w_n\), the classifier estimates the probability of each category given those words. It then predicts the category with the highest posterior probability.

The *naive* assumption is that words occur independently of one another, conditional on the document's category. Although this assumption is not strictly true for natural language, it makes the model relatively simple and computationally efficient.

In practice, this implementation works with word frequencies and log probabilities. Taking logarithms allows the individual word contributions to be added together rather than multiplied, avoiding numerical underflow when processing longer documents.

## Implementations

The project includes the following classifier implementations.

### Standard Naive Bayes

The baseline implementation uses a Naive Bayes model. Class priors and word likelihoods are estimated from the training data, with Laplace smoothing to avoid zero probabilities.

Predictions are made by maximising the log-posterior probability of each class.

### Stop-word Removal

Stop words are common words (such as "the", "and", and "of") that generally provide limited information for distinguishing between document categories.

The experiments investigate the effect of removing stop words during preprocessing. This allows the performance of models trained using the full vocabulary to be compared with models trained after removing these common words.

### Term Frequency Transformation

The multinomial Naive Bayes model uses word frequencies to estimate the likelihood of a document belonging to each class. However, raw word frequencies can give disproportionately high importance to words that occur repeatedly within a document. To reduce this effect, a logarithmic transformation is applied to each word frequency. If \(f\) is the original frequency, the transformed frequency is:

\[
f' = \log(1+f)
\]

This transformation reduces the influence of repeated occurrences of the same word while retaining information about its frequency.

### Document Frequency Transformation

Words that appear in many documents are often less useful for distinguishing between classes. Conversely, words that appear in very few documents are likely to give a better indication of that documents class. An inverse document frequency weighting is therefore applied:

\[
f' = f \cdot \frac{N}{df(w)}
\]

where \(N\) is the number of documents and \(df(w)\) is the number of documents containing word \(w\).

### Combined TF-IDF Transformation

The logarithmic term-frequency and inverse document-frequency transformations are combined to give:

\[
f' = \log(1+f)\cdot\frac{N}{df(w)}
\]

This aims to reduce the influence of frequently repeated words while giving greater weight to words that are more informative across the document collection.

### Complement Naive Bayes

Complement Naive Bayes estimates word likelihoods using documents belonging to all classes other than the class being considered.

This approach is particularly useful when working with imbalanced datasets, as it can reduce the bias towards more frequently represented classes.

The project also implements document-frequency and combined frequency transformations for Complement Naive Bayes.

## Experimental Methodology

Model performance was evaluated using stratified 10-fold cross-validation. Stratification ensures that each fold maintains approximately the same class distribution as the overall dataset.

For each fold:

1. The data is divided into training and validation sets.
2. Vocabulary selection and preprocessing are performed using the training partition.
3. The classifier is trained on the processed training data.
4. Classification accuracy is measured on the held-out validation partition.

The reported performance is the mean accuracy across the ten folds. Standard deviation is also recorded to indicate variation between folds.

Experiments investigate different vocabulary sizes, ranging from 1,000 to 20,000 words, alongside the various classifier and weighting approaches.

## Results

The following table summarises the cross-validated classification accuracy of the models. It includes the baseline results and only those alternative configurations that improved upon the corresponding standard Naive Bayes model.

| Classifier | Vocabulary size | Mean accuracy |
|---|---:|---:|
| Standard Naive Bayes | 1,000 | 90.30% |
| Standard Naive Bayes | 5,000 | 94.50% |
| Document Frequency Naive Bayes | 5,000 | 95.50% |
| Standard Naive Bayes | 15,000 | 95.48% |
| Document Frequency Naive Bayes | 15,000 | 96.25% |
| TF-IDF Naive Bayes | 15,000 | 96.58% |
| TF-IDF Naive Bayes | 20,000 | **96.85%** |

The best-performing configuration was the combined TF-IDF Naive Bayes classifier using a vocabulary of 20,000 words, achieving a mean cross-validation accuracy of 96.85%.

### Discussion

Several observations emerged from the experiments:

- Increasing vocabulary size produced substantial improvements in classification accuracy.
- Removing common stop words provided a small improvement over the equivalent standard Naive Bayes model.
- Logarithmic term-frequency weighting alone did not improve performance over the baseline in the initial experiments.
- Document-frequency weighting improved classification accuracy, suggesting that less common words provided useful discriminative information.
- Combining term-frequency and document-frequency transformations produced the strongest overall results.
- Complement Naive Bayes provided an alternative approach to handling class imbalance, although it did not outperform the best standard Naive Bayes configuration.

Overall, the experiments demonstrated that relatively simple modifications to feature representation and model estimation can substantially improve text classification performance.

## Project Structure

```text
naive-bayes-text-classification/
│
├── README.md
├── notebooks
│   ├──experiments.ipynb
│
│
├── src/
│   ├── __init__.py
│   ├── data.py
│   ├── instances.py
│   ├── preprocessing.py
│   ├── classifiers.py
│   └── evaluation.py
│
└── data/
```

The implementation is separated into modules according to responsibility:

- `data.py` – loading and writing CSV data.
- `instances.py` – custom representations of individual documents and datasets.
- `preprocessing.py` – vocabulary selection and text preprocessing.
- `classifiers.py` – Naive Bayes and Complement Naive Bayes implementations.
- `evaluation.py` – cross-validation and classification metrics.
- `experiments.ipynb` – experimental workflow and comparison of classifier performance.

## Getting Started

### Installation

Clone the repository:

```bash
git clone <repository-url>
cd naive-bayes-text-classification
```

### Dataset

The project uses a labelled collection of text abstracts for training and evaluation. The training and test data have been included in the `data/` directory.


## References

[1] Rennie, J. D. M., Shih, L., Teevan, J., & Karger, D. R. (2003). *Tackling the Poor Assumptions of Naive Bayes Text Classifiers*. Proceedings of the 20th International Conference on Machine Learning (ICML).

