# ML_DL Assignment 1 - Explanation and Solution

## Overview of the Assignment
This assignment focuses on understanding and implementing machine learning techniques, specifically:
1. **Linear Regression**:
   - Solve a linear regression problem using the least squares method.
   - Understand the coefficients and intercept of the regression line.
2. **Classification**:
   - Solve a classification problem by training a model and evaluating its accuracy.
   - Understand the classification coefficients, intercept, and class labels.

### Objectives:
- Implement mathematical solutions like the Least Squares method.
- Work with numpy to handle data and compute solutions efficiently.
- Evaluate model performance using metrics like classification accuracy.

## Solution Description

### 1. Linear Regression
**Task:** Solve a linear regression problem where \( X \theta = y \).  
**Solution:** 
- The `LeastSquares(X, y)` function calculates the coefficients \( \theta \) using the equation:
  \[
  \theta = (X^T X)^{-1} X^T y
  \]
- The coefficients represent the weights of each feature, and the intercept aligns the regression line with the data.

**Key Functions for Linear Regression:**
- **`LeastSquares(X, y)`**: Computes the least squares solution.
- **`linear_regression_coeff_submission()`**: Returns the predefined coefficients obtained from training or analysis.
- **`linear_regression_intrcpt_submission()`**: Returns the predefined intercept value.

---

### 2. Classification
**Task:** Solve a classification problem and evaluate the model's accuracy.  
**Solution:** 
- A classification model predicts labels for the input data \( X \).
- Accuracy is calculated as the ratio of correctly classified samples to the total samples:
  \[
  \text{Accuracy} = \frac{\text{Correct Classifications}}{\text{Total Samples}} \times 100
  \]
- Predefined coefficients and intercept values are used to evaluate the classification model.

**Key Functions for Classification:**
- **`classification_accuracy(model, X, s)`**: Calculates the accuracy of a classification model.
- **`classification_coeff_submission()`**: Returns the predefined coefficients for the classification problem.
- **`classification_intrcpt_submission()`**: Returns the intercept values for the classification problem.
- **`classification_classes_submission()`**: Returns the class labels for the classification problem.

### How the Problems Were Solved:
1. **Linear Regression**:
   - The Least Squares method is implemented to calculate coefficients.
   - The coefficients and intercept are extracted and saved for submission.

2. **Classification**:
   - A model is assumed to predict values for \( X \).
   - The accuracy of the model is evaluated using predefined coefficients, intercept, and class labels.

---

### Summary:
This assignment demonstrates the practical application of machine learning techniques like regression and classification. By implementing mathematical models and using predefined coefficients, the assignment reinforces fundamental concepts and their use in real-world scenarios.
