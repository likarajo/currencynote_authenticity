# Machine Learning Classifcation

---

## Decision Tree

* A **supervised** machine learning algorithm.
* Can perform both **regression** and **classification**.
  * Used to predict both continuous and discrete values.
* Advantages of Decision Trees:
  * Require relatively less effort for training the algorithm.
  * Can be used to **classify non-linearly separable data**.
  * Very fast and efficient compared to other classification algorithms.

### Decision Tree Algorithm

* **For each attribute in the dataset form a node**, where the most important attribute is placed at the root node.  
* For evaluation, start at the root node and **work the way down the tree** by following the corresponding **node that meets the condition** or *decision*.
* Continues the process until a leaf node is reached.
* The leaf node contains the prediction or the outcome of the decision tree.

---

## Ensemble Learning

* A type of learning where you **join different types of algorithms or same algorithm multiple times**.
* Forms a more powerful prediction model.  

## Random Forest

* A type of **supervised** machine learning algorithm.
* Based on **Ensemble** learning.
  * Combines multiple ***Decision trees***, resulting in a *forest of trees*.
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

### Random Forest Algorithm

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

## Support Vector Machine

* In case of linearly separable data in two dimensions, an SVM tries to **find a boundary that divides the data in such a way that the misclassification error can be minimized**.  
* SVM differs from the other classification algorithms in the way that **it chooses the decision boundary that maximizes the distance from the nearest data points of all the classes**.  
* An SVM doesn't merely find a decision boundary; it finds the most optimal decision boundary.
* The nearest points from the decision boundary that maximize the distance between the decision boundary and the points are called **support vectors**.  
