# Cancer-Diagnosis
This project aims to recognise if a breast tumour is malignant or benign given as input its shape, texture and size. The main library used is scikit-learn

## Functionality
This is a type of classification problem, which makes use of supervised learning in order to come up with a prediction. The learning set of the system is 75% of the entire dataset meaning that this labelled data will be used for the algorithm to learn to recognise patterns that are common for malignant or benign tumours and 25% will be used to test the predictions with a 10-fold cross validation. It uses a Random Forest classifier in order to predict outputs, although KNN and Stacking have also been tested.

## Processes
The dataset used for this problem was provided by the University of Wisconsin in 1995 and it contains data for 569 unique instances. The total input attributes are 30 excluding the ID number of each instance and they are the mean, the standard error and “worst” (mean of the three largest values) of the following features:
a.	radius (mean of distances from center to points on the perimeter)
b.	texture (standard deviation of gray-scale values)
c.	perimeter
d.	area
e.	smoothness (local variation in radius lengths)
f.	compactness (perimeter^2 / area - 1.0)
g.	concavity (severity of concave portions of the contour)
h.	concave points (number of concave portions of the contour)
i.	symmetry
j.	fractal dimension ("coastline approximation" - 1)
The values were collected by interactive image processing techniques and refer to the size, shape and texture of individual cells and cell clumps. The output is either M (malignant) or B (benign) and the class distribution is 212 malignant and 357 benign (Street et al., 1992). This means that the level of imbalance of the dataset is relatively low and will not affect the functionality of the classifiers.

## Algorithm Selection

The first algorithm is a K-Nearest Neighbours or KNN (Sci-kit learn, 2020a). This classifier is as powerful as simple, the training is trivial and it works with any number of classes, making it very versatile and suitable for this dataset. Compared to neural networks, it performs well with relatively small datasets, such as this one, so it was preferred over a Multi-Layer Perceptron.

A Random Forest was then tested as this algorithm can sometimes provide higher accuracy then KNN; in fact, it does not take into consideration its neighbours in a dataset, which could happen to be biased, but it transforms the data into multiple tree representations (Sci-kit learn, 2020b). This is an improvement of the Decision Tree algorithm, which was not tested for this system as different splits of the training data can lead this algorithm to create very different trees, resulting in poor performance (Sci-kit learn, 2020c). 
However, Decision Tree was selected alongside KNN and Random Forest as an estimator for the last algorithm, a Stacking classifier, which normally results in a higher prediction accuracy than each estimator alone (Sci-kit learn, 2020d). 

A Grid Search was used in order to find the parameters of each algorithm that would best suit the dataset (Sci-kit learn, 2020e). A high number of neighbours for KNN likely leads to a bias as the neighbours would be too distant in the data frame to be part of the right class, so the values selected in the Grid Search ranged from 3 to 15. On the other hand, the values of the maximum depth for Random Forest ranged from none to 10 and the number of estimators from 100 to 500.
Stacking did not need a grid search to find its best parameters as these were easily computable manually– however its estimators inherited the parameters selected for KNN and Random Forest. The final estimator used to combine the base estimators is a logistic regressor.  

## Evaluation
The metrics that were used to evaluate the algorithms are accuracy, precision, recall and F1 score.

In order to avoid overfitting, which occurs when the metrics are tested always on the same data and ignore the results of the yet-unseen data, a 10-fold cross validation from the scikit-learn library was used. The 10-fold cross validation was run 5 times in order to factor anomalous results out. When annotated, the results were rounded to 3 significant figures and then the average of each field was calculated.

![image](https://user-images.githubusercontent.com/62818869/124388835-8491de00-dcdc-11eb-9c55-5342b563b241.png)
The graph shows that Random Forest (RF) is the most accurate model and that Stacking (ST) performs better than KNN

![image](https://user-images.githubusercontent.com/62818869/124388853-92dffa00-dcdc-11eb-9b0e-94717f32a2a2.png)
Although ST performed better than Random Forest in 2 tests out of 3, RF is the most precise model.

![image](https://user-images.githubusercontent.com/62818869/124388862-9f645280-dcdc-11eb-9e79-376a400e6b4f.png)
RF performed the best recall in 4 tests out of 5. As for the other metrics, KNN performed the worst in every test.

![image](https://user-images.githubusercontent.com/62818869/124388876-ac814180-dcdc-11eb-8ee9-64bcfe97b6b3.png)
The difference of F1-score between Random Forest and Stacking is not significant – however this is not enough to prefer ST over RF as the optimal classifier.
Below are the averages of each field compared by metric:

![image](https://user-images.githubusercontent.com/62818869/124388887-bc008a80-dcdc-11eb-9264-26d3da7e0531.png)
![image](https://user-images.githubusercontent.com/62818869/124388894-c1f66b80-dcdc-11eb-84be-cd0c283c185d.png)
![image](https://user-images.githubusercontent.com/62818869/124388895-c753b600-dcdc-11eb-80d4-5dc1e99588b7.png)
![image](https://user-images.githubusercontent.com/62818869/124388901-cb7fd380-dcdc-11eb-9e1d-0661b4958a3e.png)

## Conclusion
A Random Forest classifier was selected over K-Nearest Neighbours and Stacking as the optimal classifier for this system. This means that the model can predict whether a breast tumour is malignant or benign with an accuracy of approximately 97% and the high precision, recall and F1-score support the high reliability of the algorithm, meaning that the system is successful.

