# Currency Note Authenticity

Predict whether a bank currency note is authentic or not based on variance, skewness, and curtosis of the wavelet transformed image, and entropy of the image using **Random Forest classifier**.

---

## Background

### Ensemble Learning

* A type of learning where you **join different types of algorithms or same algorithm multiple times**.
* Forms a more powerful prediction model.  

### Random Forest

* A type of **supervised** machine learning algorithm.
* Based on **Ensemble** learning.
  * Combines multiple *Decision trees*, resulting in a *forest of trees*.
* Can be used for both regression and classification tasks.
* ***Advantages***:
  * It is **not biased**, since, there are multiple trees and each of which are trained on a subset of data. It relies on the power of *the crowd*, and so the overall biasedness of the algorithm is reduced.
  * It is **very stable**. Even if a new data point is introduced in the dataset the overall algorithm is not affected much since new data may impact one tree, but it is very hard for it to impact all the trees.
  * It **works well for both categorical and numerical features**.
  * It **works well when data has missing values**.
  * It **works well even if the data is not scaled well**.
* ***Disadvantages***:
  * It has **high complexity** requiring much **more computational resources**, owing to the large number of decision trees joined together.
  * Due to their complexity, they **require much more time to train** than other comparable algorithms.  

### Working of the Random Forest Algorithm

* Pick N random records from the dataset.
* Build a decision tree based on these N records.
* Choose the number of trees you want in your algorithm and repeat steps 1 and 2.
* In case of a regression problem:
  * For a new record, each tree in the forest predicts a value for Y (output).  
  * The final value can be calculated by taking the average of all the values predicted by all the trees in forest.  
* In case of a classification problem:
  * Each tree in the forest predicts the category to which the new record belongs.  
  * The final category that is assigned is the category that wins the majority vote.

---

## Goal

* Binary Classification problem.
* Use a **Random Forest Classifier** to solve.  

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

## Create and Train model

* Use **Random Forest Classifier**
* **Grid search with Cross Validation** for training models with different hyperparameter values
* Select the best model with the best accuracy

```
Best parameters:
 {'bootstrap': True, 'criterion': 'gini', 'n_estimators': 20}

Best training accuracy: 0.9927023661270237
```

## Evaluate the model

* Accuracy score = 0.9890909

* Confusion matrix -> only 3 errors

```
[[155   2]
 [  1 117]]
```

* Classification report

```
               precision    recall  f1-score   support

           0       0.99      0.99      0.99       157
           1       0.98      0.99      0.99       118

    accuracy                           0.99       275
   macro avg       0.99      0.99      0.99       275
weighted avg       0.99      0.99      0.99       275
```
