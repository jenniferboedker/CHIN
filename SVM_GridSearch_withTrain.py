# Script to generate and train a support vector machine (SVM) with a training data set
# and to predict activities of molecules from a test data set

# by Jennifer Bödker, Tobias Nietsch

import matplotlib.pyplot as plt
import pandas as pd
import argparse
from sklearn import svm
from sklearn.metrics import accuracy_score, recall_score, matthews_corrcoef
from sklearn.metrics import roc_curve, auc, confusion_matrix
from sklearn.model_selection import GridSearchCV
#from sklearn.model_selection import StratifiedShuffleSplit



# Parameter handling
parser = argparse.ArgumentParser(description='Bio-activity predictor (SVM)')
# mandatory input parameters
parser.add_argument('-train', type=str, required=True,
                    help='path to training data (comma separated smile, activity, list of features)')
parser.add_argument('-test', type=str, required=True,
                    help='path to test data (comma separated smile, activity, list of features)')
parser.add_argument('-out', type=str, required=True, help='path for output file (csv)')

args = parser.parse_args()
data_test= args.test
data_train = args.train
outfile = args.out


dataframe_train = pd.read_csv(data_train)
dataset_train = dataframe_train.values          # matrix like

dataframe_test = pd.read_csv(data_test)
dataset_test = dataframe_test.values


# x is input, y is output
train_x = dataset_train[:, 2:121].astype(float) # for each molecule/row use the feature info from col 2-119
train_y = dataset_train[:, 0].astype(int)
train_s = dataset_train[:, 1]                   #smiles

test_x = dataset_test[:, 2:121].astype(float)
test_y = dataset_test[:, 0].astype(int)
test_s = dataset_test[:, 1]


# Configuring the SVM predictor and the grid
estimator = svm.SVC(probability=True)

kernel = ['rbf']
#kernel = ['rbf', 'poly', 'linear']
C = [1, 10, 100, 1000, 10000]
gamma = [1e-7, 5e-7, 9e-7, 1e-8, 5e-8]
param_grid = dict(kernel=kernel, C=C, gamma=gamma)
#cv = StratifiedShuffleSplit(n_splits=10, test_size=0.5, random_state=20)

grid = GridSearchCV(estimator=estimator, param_grid=param_grid, n_jobs=1, scoring='roc_auc', cv=5)

# Train the predictor
grid.fit(train_x, train_y)

# Predicted activities
pred_train = list(grid.predict(train_x))
pred_test = list(grid.predict(test_x))

# Performance measures
print("Train Accuracy   : ", accuracy_score(train_y, grid.predict(train_x)))
print("Test Accuracy    : ", accuracy_score(test_y, pred_test))
print("Train Sensitivity: ", recall_score(train_y, pred_train))
print("Train MCC        : ", matthews_corrcoef(train_y, pred_train))
print("Test Sensitivity : ", recall_score(test_y, pred_test))
print("Test MCC         : ", matthews_corrcoef(test_y, pred_test))
print("Confusion matrix : ", confusion_matrix(test_y, pred_test)) # [[tp, fp] [fn, tn]]

# Plotting the ROC curve
fpr, tpr, _ = roc_curve(test_y, grid.predict_proba(test_x)[:,1])

auc_score = auc(fpr, tpr)

plt.title('ROC Curve')
plt.plot(fpr, tpr, label='AUC = {:.2f}'.format(auc_score), color='red')
plt.plot([0, 1], [0, 1], color='grey', linestyle=':')

plt.show()

#write outFile to evaluate
with open(outfile, 'w') as out:
    #write train data
    for line in range(0, len(pred_train)):
        out.write(str(train_s[line]) + "," + str(list(train_y)[line]) + "," + str(pred_train[line]) + "\n")
    #write test data
    for line in range(0, len(pred_test)):
        out.write(str(test_s[line]) + "," + str(list(test_y)[line]) + "," + str(pred_test[line]) + "\n")
