# script to generate a neuronal random forest for a given data set
# by Jennifer Bödker, Tobias Nietsch

import argparse
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, matthews_corrcoef
import matplotlib.pyplot as plt
import sklearn.metrics

# Parameter handeling
parser = argparse.ArgumentParser(description='Clustering of Fingerprints')
# mandatory input parameters
parser.add_argument('-t', type=str, required=True,
                    help='path to training data (comma separated smile, activity, list of features)')
parser.add_argument('-o', type=str, required=True, help='path for output file (csv)')

args = parser.parse_args()
data = args.t
outfile = args.o

# fix random seed for reproducibility
seed = 7
np.random.seed(seed)

dataframe = pd.read_csv(data)
dataset = dataframe.values  # matrix like

# x is input, y is output
x = dataset[:, 2:121].astype(float)  # for each molecule/row use the feature info from col 2-119
y = dataset[:, 0].astype(int)  # todo need the cast to int???
s = dataset[:, 1] #smiles


# split data to train and test model
train_x, test_x, train_y, test_y, train_s, test_s = train_test_split(x, y,s, shuffle=False,
                                                    train_size=0.7)  # 0.7 in example as standard

estimator = RandomForestClassifier()
# Number of trees in random forest
n_estimators = [int(x) for x in np.linspace(start = 100, stop = 2000, num = 10)] #1000 - 0.83, 2000 - 0.84
# Number of features to consider at every split
max_features = [119, 60, 40]
# Method of selecting samples for training each tree
bootstrap = [True, False]
#criterion for calculating the decision trees
criterion = ['gini', 'entropy']

param_grid = dict(n_estimators=n_estimators,max_features=max_features,bootstrap=bootstrap,criterion=criterion)
print(param_grid)

# GridSearch cannot run in parallel
#specify cv?
grid = GridSearchCV(estimator=estimator, param_grid=param_grid, n_jobs=1, scoring='roc_auc',cv=5)

grid.fit(train_x, train_y)
pred_train = list(grid.predict(train_x))
pred_test = list(grid.predict(test_x))

print("Train Accuracy :: ", accuracy_score(train_y, grid.predict(train_x)))
print("Test Accuracy  :: ", accuracy_score(test_y, pred_test))
print("Confusion matrix test", confusion_matrix(test_y, pred_test)) # tn, fp, fn, tp
print("Confusion matrix train", confusion_matrix(train_y, pred_train)) # tn, fp, fn, tp
print("MCC  :: ", matthews_corrcoef(test_y, pred_test))
print("MCC  :: ", matthews_corrcoef(train_y, pred_train))

#####################plots

fpr, tpr, _ = sklearn.metrics.roc_curve(train_y, grid.predict_proba(train_x)[:,1])

auc_score = sklearn.metrics.auc(fpr, tpr)

plt.plot(fpr, tpr, label='AUC = {:.2f}'.format(auc_score), color='red')
plt.plot([0, 1], [0, 1], color='grey', linestyle=':')
plt.title('ROC Curve')

plt.show()

#########################

#write outFile to evaluate
with open(outfile, 'w') as out:
    #out.write("SMILES,Bio-activity,Predicted bio-activity \n")
    #write test data
    for line in range(0, len(pred_train)):
        out.write(str(train_s[line])+","+str(list(train_y)[line]) + "," + str(pred_train[line]) + "\n")
    #write train data
    for line in range(0, len(pred_test)):
        out.write(str(test_s[line])+","+str(list(test_y)[line]) + "," + str(pred_test[line]) + "\n")

