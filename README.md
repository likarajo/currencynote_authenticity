# Currency Note Authenticity

Predict whether a bank currency note is authentic or not based on variance, skewness, and curtosis of the wavelet transformed image, and entropy of the image using classifiers.

## Background

Machine Learning classifiers: [classifiers.md](classifiers.md)

## Goal

* Binary Classification problem.
* Use a **Decision Tree Classifier**.
* Use a **Random Forest Classifier**.  
* Use a **Support Vector Machine**.

## Dependencies

* Pandas
* Scikit-learn

`pip install -r requirements.txt`

## Dataset

Bank note authetication dataset from UCI archive: https://archive.ics.uci.edu/ml/datasets/banknote+authentication<br>
Saved in: *data/banknote_authentication.txt*

1. **Variance** of Wavelet Transformed image (continuous)
2. **Skewness** of Wavelet Transformed image (continuous)
3. **Curtosis** of Wavelet Transformed image (continuous)
4. **Entropy** of image (continuous)
5. **Class** (integer)

## Data Preprocessing

* Preprare features and labels
  * Features = Variance, Skewness, Curtosis, Entropy
  * Label = Class
* Split data into Training and Test sets
  * Test size = 20%
* Scale the features

## Create, Train and Evaluate model

* Use **Cross Validation** training models with different hyperparameter values.
* Use **Grid Search**  to select the best model with the best accuracy.

### Decision Tree Classifier

[authentication_DT.ipynb](authentication_DT.ipynb)

```
Grid Search scores:

0.977 (+/-0.023) for {'criterion': 'gini'}
0.981 (+/-0.023) for {'criterion': 'entropy'}

Best parameters:
 {'criterion': 'entropy'}

Training accuracy: 98.08592777085927 %

Test Accuracy: 98.54545454545455 %

Confusion matrix:
 [[154   3]
 [  1 117]]

Classification report:
               precision    recall  f1-score   support

           0       0.99      0.98      0.99       157
           1       0.97      0.99      0.98       118

    accuracy                           0.99       275
   macro avg       0.98      0.99      0.99       275
weighted avg       0.99      0.99      0.99       275
```

### Random Forest Classifier

[authentication_RF.ipynb](authentication_RF.ipynb)

```
Grid Search scores:

0.984 (+/-0.033) for {'bootstrap': True, 'criterion': 'gini', 'n_estimators': 5}
0.986 (+/-0.019) for {'bootstrap': True, 'criterion': 'gini', 'n_estimators': 10}
0.993 (+/-0.014) for {'bootstrap': True, 'criterion': 'gini', 'n_estimators': 20}
0.991 (+/-0.011) for {'bootstrap': True, 'criterion': 'gini', 'n_estimators': 40}
0.985 (+/-0.018) for {'bootstrap': True, 'criterion': 'entropy', 'n_estimators': 5}
0.986 (+/-0.016) for {'bootstrap': True, 'criterion': 'entropy', 'n_estimators': 10}
0.986 (+/-0.016) for {'bootstrap': True, 'criterion': 'entropy', 'n_estimators': 20}
0.985 (+/-0.023) for {'bootstrap': True, 'criterion': 'entropy', 'n_estimators': 40}
0.985 (+/-0.022) for {'bootstrap': False, 'criterion': 'gini', 'n_estimators': 5}
0.989 (+/-0.012) for {'bootstrap': False, 'criterion': 'gini', 'n_estimators': 10}
0.992 (+/-0.007) for {'bootstrap': False, 'criterion': 'gini', 'n_estimators': 20}
0.991 (+/-0.015) for {'bootstrap': False, 'criterion': 'gini', 'n_estimators': 40}
0.988 (+/-0.016) for {'bootstrap': False, 'criterion': 'entropy', 'n_estimators': 5}
0.989 (+/-0.014) for {'bootstrap': False, 'criterion': 'entropy', 'n_estimators': 10}
0.989 (+/-0.014) for {'bootstrap': False, 'criterion': 'entropy', 'n_estimators': 20}
0.988 (+/-0.012) for {'bootstrap': False, 'criterion': 'entropy', 'n_estimators': 40}

Best parameters:
 {'bootstrap': True, 'criterion': 'gini', 'n_estimators': 20}

Training accuracy: 99.27023661270236 %

Test Accuracy: 98.9090909090909 %

Confusion matrix:
 [[155   2]
 [  1 117]]

Classification report:
               precision    recall  f1-score   support

           0       0.99      0.99      0.99       157
           1       0.98      0.99      0.99       118

    accuracy                           0.99       275
   macro avg       0.99      0.99      0.99       275
weighted avg       0.99      0.99      0.99       275
```

### Support Vector Machine

[authentication_SVM.ipynb](authentication_SVM.ipynb)

```
Grid Search scores:

0.974 (+/-0.021) for {'C': 0.01, 'degree': 1, 'gamma': 'auto', 'kernel': 'linear'}
0.941 (+/-0.027) for {'C': 0.01, 'degree': 1, 'gamma': 'auto', 'kernel': 'poly'}
0.924 (+/-0.019) for {'C': 0.01, 'degree': 1, 'gamma': 'auto', 'kernel': 'rbf'}
0.907 (+/-0.029) for {'C': 0.01, 'degree': 1, 'gamma': 'auto', 'kernel': 'sigmoid'}
0.974 (+/-0.021) for {'C': 0.01, 'degree': 1, 'gamma': 'scale', 'kernel': 'linear'}
0.941 (+/-0.027) for {'C': 0.01, 'degree': 1, 'gamma': 'scale', 'kernel': 'poly'}
0.923 (+/-0.021) for {'C': 0.01, 'degree': 1, 'gamma': 'scale', 'kernel': 'rbf'}
0.907 (+/-0.029) for {'C': 0.01, 'degree': 1, 'gamma': 'scale', 'kernel': 'sigmoid'}
0.974 (+/-0.021) for {'C': 0.01, 'degree': 2, 'gamma': 'auto', 'kernel': 'linear'}
0.582 (+/-0.018) for {'C': 0.01, 'degree': 2, 'gamma': 'auto', 'kernel': 'poly'}
0.924 (+/-0.019) for {'C': 0.01, 'degree': 2, 'gamma': 'auto', 'kernel': 'rbf'}
0.907 (+/-0.029) for {'C': 0.01, 'degree': 2, 'gamma': 'auto', 'kernel': 'sigmoid'}
0.974 (+/-0.021) for {'C': 0.01, 'degree': 2, 'gamma': 'scale', 'kernel': 'linear'}
0.582 (+/-0.018) for {'C': 0.01, 'degree': 2, 'gamma': 'scale', 'kernel': 'poly'}
0.923 (+/-0.021) for {'C': 0.01, 'degree': 2, 'gamma': 'scale', 'kernel': 'rbf'}
0.907 (+/-0.029) for {'C': 0.01, 'degree': 2, 'gamma': 'scale', 'kernel': 'sigmoid'}
0.974 (+/-0.021) for {'C': 0.01, 'degree': 3, 'gamma': 'auto', 'kernel': 'linear'}
0.715 (+/-0.023) for {'C': 0.01, 'degree': 3, 'gamma': 'auto', 'kernel': 'poly'}
0.924 (+/-0.019) for {'C': 0.01, 'degree': 3, 'gamma': 'auto', 'kernel': 'rbf'}
0.907 (+/-0.029) for {'C': 0.01, 'degree': 3, 'gamma': 'auto', 'kernel': 'sigmoid'}
0.974 (+/-0.021) for {'C': 0.01, 'degree': 3, 'gamma': 'scale', 'kernel': 'linear'}
0.714 (+/-0.025) for {'C': 0.01, 'degree': 3, 'gamma': 'scale', 'kernel': 'poly'}
0.923 (+/-0.021) for {'C': 0.01, 'degree': 3, 'gamma': 'scale', 'kernel': 'rbf'}
0.907 (+/-0.029) for {'C': 0.01, 'degree': 3, 'gamma': 'scale', 'kernel': 'sigmoid'}
0.974 (+/-0.021) for {'C': 0.01, 'degree': 4, 'gamma': 'auto', 'kernel': 'linear'}
0.649 (+/-0.028) for {'C': 0.01, 'degree': 4, 'gamma': 'auto', 'kernel': 'poly'}
0.924 (+/-0.019) for {'C': 0.01, 'degree': 4, 'gamma': 'auto', 'kernel': 'rbf'}
0.907 (+/-0.029) for {'C': 0.01, 'degree': 4, 'gamma': 'auto', 'kernel': 'sigmoid'}
0.974 (+/-0.021) for {'C': 0.01, 'degree': 4, 'gamma': 'scale', 'kernel': 'linear'}
0.649 (+/-0.028) for {'C': 0.01, 'degree': 4, 'gamma': 'scale', 'kernel': 'poly'}
0.923 (+/-0.021) for {'C': 0.01, 'degree': 4, 'gamma': 'scale', 'kernel': 'rbf'}
0.907 (+/-0.029) for {'C': 0.01, 'degree': 4, 'gamma': 'scale', 'kernel': 'sigmoid'}
0.980 (+/-0.016) for {'C': 0.1, 'degree': 1, 'gamma': 'auto', 'kernel': 'linear'}
0.978 (+/-0.018) for {'C': 0.1, 'degree': 1, 'gamma': 'auto', 'kernel': 'poly'}
0.988 (+/-0.018) for {'C': 0.1, 'degree': 1, 'gamma': 'auto', 'kernel': 'rbf'}
0.854 (+/-0.029) for {'C': 0.1, 'degree': 1, 'gamma': 'auto', 'kernel': 'sigmoid'}
0.980 (+/-0.016) for {'C': 0.1, 'degree': 1, 'gamma': 'scale', 'kernel': 'linear'}
0.978 (+/-0.018) for {'C': 0.1, 'degree': 1, 'gamma': 'scale', 'kernel': 'poly'}
0.988 (+/-0.018) for {'C': 0.1, 'degree': 1, 'gamma': 'scale', 'kernel': 'rbf'}
0.856 (+/-0.027) for {'C': 0.1, 'degree': 1, 'gamma': 'scale', 'kernel': 'sigmoid'}
0.980 (+/-0.016) for {'C': 0.1, 'degree': 2, 'gamma': 'auto', 'kernel': 'linear'}
0.603 (+/-0.035) for {'C': 0.1, 'degree': 2, 'gamma': 'auto', 'kernel': 'poly'}
0.988 (+/-0.018) for {'C': 0.1, 'degree': 2, 'gamma': 'auto', 'kernel': 'rbf'}
0.854 (+/-0.029) for {'C': 0.1, 'degree': 2, 'gamma': 'auto', 'kernel': 'sigmoid'}
0.980 (+/-0.016) for {'C': 0.1, 'degree': 2, 'gamma': 'scale', 'kernel': 'linear'}
0.603 (+/-0.035) for {'C': 0.1, 'degree': 2, 'gamma': 'scale', 'kernel': 'poly'}
0.988 (+/-0.018) for {'C': 0.1, 'degree': 2, 'gamma': 'scale', 'kernel': 'rbf'}
0.856 (+/-0.027) for {'C': 0.1, 'degree': 2, 'gamma': 'scale', 'kernel': 'sigmoid'}
0.980 (+/-0.016) for {'C': 0.1, 'degree': 3, 'gamma': 'auto', 'kernel': 'linear'}
0.932 (+/-0.045) for {'C': 0.1, 'degree': 3, 'gamma': 'auto', 'kernel': 'poly'}
0.988 (+/-0.018) for {'C': 0.1, 'degree': 3, 'gamma': 'auto', 'kernel': 'rbf'}
0.854 (+/-0.029) for {'C': 0.1, 'degree': 3, 'gamma': 'auto', 'kernel': 'sigmoid'}
0.980 (+/-0.016) for {'C': 0.1, 'degree': 3, 'gamma': 'scale', 'kernel': 'linear'}
0.933 (+/-0.033) for {'C': 0.1, 'degree': 3, 'gamma': 'scale', 'kernel': 'poly'}
0.988 (+/-0.018) for {'C': 0.1, 'degree': 3, 'gamma': 'scale', 'kernel': 'rbf'}
0.856 (+/-0.027) for {'C': 0.1, 'degree': 3, 'gamma': 'scale', 'kernel': 'sigmoid'}
0.980 (+/-0.016) for {'C': 0.1, 'degree': 4, 'gamma': 'auto', 'kernel': 'linear'}
0.680 (+/-0.024) for {'C': 0.1, 'degree': 4, 'gamma': 'auto', 'kernel': 'poly'}
0.988 (+/-0.018) for {'C': 0.1, 'degree': 4, 'gamma': 'auto', 'kernel': 'rbf'}
0.854 (+/-0.029) for {'C': 0.1, 'degree': 4, 'gamma': 'auto', 'kernel': 'sigmoid'}
0.980 (+/-0.016) for {'C': 0.1, 'degree': 4, 'gamma': 'scale', 'kernel': 'linear'}
0.680 (+/-0.024) for {'C': 0.1, 'degree': 4, 'gamma': 'scale', 'kernel': 'poly'}
0.988 (+/-0.018) for {'C': 0.1, 'degree': 4, 'gamma': 'scale', 'kernel': 'rbf'}
0.856 (+/-0.027) for {'C': 0.1, 'degree': 4, 'gamma': 'scale', 'kernel': 'sigmoid'}
0.984 (+/-0.017) for {'C': 1.0, 'degree': 1, 'gamma': 'auto', 'kernel': 'linear'}
0.982 (+/-0.019) for {'C': 1.0, 'degree': 1, 'gamma': 'auto', 'kernel': 'poly'}
1.000 (+/-0.000) for {'C': 1.0, 'degree': 1, 'gamma': 'auto', 'kernel': 'rbf'}
0.785 (+/-0.050) for {'C': 1.0, 'degree': 1, 'gamma': 'auto', 'kernel': 'sigmoid'}
0.984 (+/-0.017) for {'C': 1.0, 'degree': 1, 'gamma': 'scale', 'kernel': 'linear'}
0.982 (+/-0.019) for {'C': 1.0, 'degree': 1, 'gamma': 'scale', 'kernel': 'poly'}
1.000 (+/-0.000) for {'C': 1.0, 'degree': 1, 'gamma': 'scale', 'kernel': 'rbf'}
0.785 (+/-0.046) for {'C': 1.0, 'degree': 1, 'gamma': 'scale', 'kernel': 'sigmoid'}
0.984 (+/-0.017) for {'C': 1.0, 'degree': 2, 'gamma': 'auto', 'kernel': 'linear'}
0.762 (+/-0.025) for {'C': 1.0, 'degree': 2, 'gamma': 'auto', 'kernel': 'poly'}
1.000 (+/-0.000) for {'C': 1.0, 'degree': 2, 'gamma': 'auto', 'kernel': 'rbf'}
0.785 (+/-0.050) for {'C': 1.0, 'degree': 2, 'gamma': 'auto', 'kernel': 'sigmoid'}
0.984 (+/-0.017) for {'C': 1.0, 'degree': 2, 'gamma': 'scale', 'kernel': 'linear'}
0.760 (+/-0.021) for {'C': 1.0, 'degree': 2, 'gamma': 'scale', 'kernel': 'poly'}
1.000 (+/-0.000) for {'C': 1.0, 'degree': 2, 'gamma': 'scale', 'kernel': 'rbf'}
0.785 (+/-0.046) for {'C': 1.0, 'degree': 2, 'gamma': 'scale', 'kernel': 'sigmoid'}
0.984 (+/-0.017) for {'C': 1.0, 'degree': 3, 'gamma': 'auto', 'kernel': 'linear'}
0.986 (+/-0.021) for {'C': 1.0, 'degree': 3, 'gamma': 'auto', 'kernel': 'poly'}
1.000 (+/-0.000) for {'C': 1.0, 'degree': 3, 'gamma': 'auto', 'kernel': 'rbf'}
0.785 (+/-0.050) for {'C': 1.0, 'degree': 3, 'gamma': 'auto', 'kernel': 'sigmoid'}
0.984 (+/-0.017) for {'C': 1.0, 'degree': 3, 'gamma': 'scale', 'kernel': 'linear'}
0.986 (+/-0.021) for {'C': 1.0, 'degree': 3, 'gamma': 'scale', 'kernel': 'poly'}
1.000 (+/-0.000) for {'C': 1.0, 'degree': 3, 'gamma': 'scale', 'kernel': 'rbf'}
0.785 (+/-0.046) for {'C': 1.0, 'degree': 3, 'gamma': 'scale', 'kernel': 'sigmoid'}
0.984 (+/-0.017) for {'C': 1.0, 'degree': 4, 'gamma': 'auto', 'kernel': 'linear'}
0.768 (+/-0.036) for {'C': 1.0, 'degree': 4, 'gamma': 'auto', 'kernel': 'poly'}
1.000 (+/-0.000) for {'C': 1.0, 'degree': 4, 'gamma': 'auto', 'kernel': 'rbf'}
0.785 (+/-0.050) for {'C': 1.0, 'degree': 4, 'gamma': 'auto', 'kernel': 'sigmoid'}
0.984 (+/-0.017) for {'C': 1.0, 'degree': 4, 'gamma': 'scale', 'kernel': 'linear'}
0.764 (+/-0.034) for {'C': 1.0, 'degree': 4, 'gamma': 'scale', 'kernel': 'poly'}
1.000 (+/-0.000) for {'C': 1.0, 'degree': 4, 'gamma': 'scale', 'kernel': 'rbf'}
0.785 (+/-0.046) for {'C': 1.0, 'degree': 4, 'gamma': 'scale', 'kernel': 'sigmoid'}
0.988 (+/-0.016) for {'C': 10, 'degree': 1, 'gamma': 'auto', 'kernel': 'linear'}
0.985 (+/-0.017) for {'C': 10, 'degree': 1, 'gamma': 'auto', 'kernel': 'poly'}
1.000 (+/-0.000) for {'C': 10, 'degree': 1, 'gamma': 'auto', 'kernel': 'rbf'}
0.763 (+/-0.057) for {'C': 10, 'degree': 1, 'gamma': 'auto', 'kernel': 'sigmoid'}
0.988 (+/-0.016) for {'C': 10, 'degree': 1, 'gamma': 'scale', 'kernel': 'linear'}
0.985 (+/-0.016) for {'C': 10, 'degree': 1, 'gamma': 'scale', 'kernel': 'poly'}
1.000 (+/-0.000) for {'C': 10, 'degree': 1, 'gamma': 'scale', 'kernel': 'rbf'}
0.763 (+/-0.063) for {'C': 10, 'degree': 1, 'gamma': 'scale', 'kernel': 'sigmoid'}
0.988 (+/-0.016) for {'C': 10, 'degree': 2, 'gamma': 'auto', 'kernel': 'linear'}
0.796 (+/-0.025) for {'C': 10, 'degree': 2, 'gamma': 'auto', 'kernel': 'poly'}
1.000 (+/-0.000) for {'C': 10, 'degree': 2, 'gamma': 'auto', 'kernel': 'rbf'}
0.763 (+/-0.057) for {'C': 10, 'degree': 2, 'gamma': 'auto', 'kernel': 'sigmoid'}
0.988 (+/-0.016) for {'C': 10, 'degree': 2, 'gamma': 'scale', 'kernel': 'linear'}
0.795 (+/-0.027) for {'C': 10, 'degree': 2, 'gamma': 'scale', 'kernel': 'poly'}
1.000 (+/-0.000) for {'C': 10, 'degree': 2, 'gamma': 'scale', 'kernel': 'rbf'}
0.763 (+/-0.063) for {'C': 10, 'degree': 2, 'gamma': 'scale', 'kernel': 'sigmoid'}
0.988 (+/-0.016) for {'C': 10, 'degree': 3, 'gamma': 'auto', 'kernel': 'linear'}
0.988 (+/-0.014) for {'C': 10, 'degree': 3, 'gamma': 'auto', 'kernel': 'poly'}
1.000 (+/-0.000) for {'C': 10, 'degree': 3, 'gamma': 'auto', 'kernel': 'rbf'}
0.763 (+/-0.057) for {'C': 10, 'degree': 3, 'gamma': 'auto', 'kernel': 'sigmoid'}
0.988 (+/-0.016) for {'C': 10, 'degree': 3, 'gamma': 'scale', 'kernel': 'linear'}
0.988 (+/-0.014) for {'C': 10, 'degree': 3, 'gamma': 'scale', 'kernel': 'poly'}
1.000 (+/-0.000) for {'C': 10, 'degree': 3, 'gamma': 'scale', 'kernel': 'rbf'}
0.763 (+/-0.063) for {'C': 10, 'degree': 3, 'gamma': 'scale', 'kernel': 'sigmoid'}
0.988 (+/-0.016) for {'C': 10, 'degree': 4, 'gamma': 'auto', 'kernel': 'linear'}
0.863 (+/-0.019) for {'C': 10, 'degree': 4, 'gamma': 'auto', 'kernel': 'poly'}
1.000 (+/-0.000) for {'C': 10, 'degree': 4, 'gamma': 'auto', 'kernel': 'rbf'}
0.763 (+/-0.057) for {'C': 10, 'degree': 4, 'gamma': 'auto', 'kernel': 'sigmoid'}
0.988 (+/-0.016) for {'C': 10, 'degree': 4, 'gamma': 'scale', 'kernel': 'linear'}
0.862 (+/-0.017) for {'C': 10, 'degree': 4, 'gamma': 'scale', 'kernel': 'poly'}
1.000 (+/-0.000) for {'C': 10, 'degree': 4, 'gamma': 'scale', 'kernel': 'rbf'}
0.763 (+/-0.063) for {'C': 10, 'degree': 4, 'gamma': 'scale', 'kernel': 'sigmoid'}

Best parameters:
 {'C': 1.0, 'degree': 1, 'gamma': 'auto', 'kernel': 'rbf'}

Training accuracy: 100.0 %

Test Accuracy: 100.0 %

Confusion matrix:
 [[157   0]
 [  0 118]]

Classification report:
               precision    recall  f1-score   support

           0       1.00      1.00      1.00       157
           1       1.00      1.00      1.00       118

    accuracy                           1.00       275
   macro avg       1.00      1.00      1.00       275
weighted avg       1.00      1.00      1.00       275
```

### K Nearest Neighbors

```
Grid Search scores:

0.999 (+/-0.004) for {'n_neighbors': 5}
0.999 (+/-0.004) for {'n_neighbors': 10}
0.991 (+/-0.012) for {'n_neighbors': 20}

Best parameters:
 {'n_neighbors': 5}

Training accuracy: 99.90867579908675 %

Test Accuracy: 99.63636363636364 %

Confusion matrix:
 [[153   1]
 [  0 121]]

Classification report:
               precision    recall  f1-score   support

           0       1.00      0.99      1.00       154
           1       0.99      1.00      1.00       121

    accuracy                           1.00       275
   macro avg       1.00      1.00      1.00       275
weighted avg       1.00      1.00      1.00       275
```

