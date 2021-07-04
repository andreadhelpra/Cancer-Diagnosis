import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import ShuffleSplit
from sklearn.model_selection import cross_val_score

data = pd.read_csv("breast-cancer.csv")
print(data)
X = data.values[:, 1:30] # Input fields
Y = data.values[:, 31] # Output field

#KNN classifier
knn_clf = KNeighborsClassifier(n_neighbors=5,
                           weights='distance',
                           metric='manhattan')
# from sklearn.model_selection import GridSearchCV
# knn_params = {'n_neighbors':[3,5,10,15], 'weights':['uniform', 'distance'],
#              'metric':['euclidean','manhattan']}
# gclf = GridSearchCV(clf, knn_params, n_jobs=-1, cv=3)
# gclf.fit(X_train, Y_train)
# print('Best parameters found:\n', gclf.best_params_)

# Random Forest Classifier
rf_clf = RandomForestClassifier( max_depth=10, n_estimators= 200,
                               bootstrap=False, criterion='gini')
# rf_param = { 'max_depth':[3,5,10,None], 'n_estimators':[100,200,300,400,500],
#      'criterion':['gini', 'entropy'], 'bootstrap':[True, False]}
# gclf = GridSearchCV(clf3, rf_param, n_jobs=-1, cv=3)
# gclf.fit(X_train, Y_train)
# print('Best parameters found:\n', gclf.best_params_)

# Stacking classifier using KNN, Decision tree and Random Forest
estimators = [('kn', knn_clf), ('dt', DecisionTreeClassifier()), ('rf', rf_clf)]
sclf = StackingClassifier(estimators=estimators, final_estimator=LogisticRegression())

# 10-fold cross validation for each classifier.
# Evaluating accuracy, precision, recall, F1 score.
print("10-fold cross validation:\n")
for clf, label in zip([knn_clf, rf_clf, sclf],
                      ['KNN', 'Random Forest', 'Stacking']):
    cv = ShuffleSplit(n_splits=10, test_size=0.25)
    accuracy_scores = cross_val_score(clf, X, Y, cv=cv, scoring="accuracy")
    precision_scores = cross_val_score(clf, X, Y, cv=cv, scoring="precision_macro")
    recall_scores = cross_val_score(clf, X, Y, cv=cv, scoring="recall_macro")
    f_scores = cross_val_score(clf, X, Y, cv=cv, scoring="f1_macro")
    print(label,
          "\nAccuracy mean:", accuracy_scores.mean(),
          "\nPrecision mean:", precision_scores.mean(),
          "\nRecall mean:", recall_scores.mean(),
          "\nF1 mean:", f_scores.mean())
    print()
