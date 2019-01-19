# script to generate a neuronal network for a given data set
# by Jennifer Bödker, Tobias Nietsch

import argparse
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense
from sklearn.model_selection import GridSearchCV
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, matthews_corrcoef
from sklearn import metrics
import matplotlib.pyplot as plt

# Parameter handeling
parser = argparse.ArgumentParser(description='Clustering of Fingerprints')
# mandatory input parameters
parser.add_argument('-t', type=str, required=True,
                    help='path to training data (comma separated smile, activity, list of features)')
parser.add_argument('-o', type=str, required=True, help='path for output file (csv)')

args = parser.parse_args()
data = args.t
outfile = args.o


# parameters entered here are filled in with param_grid later
def create_baseline(epochs=1, batch_size=1):
    # create model
    model = Sequential()
    model.add(Dense(119, input_dim=119, kernel_initializer='normal', activation='relu'))
    model.add(Dense(60, kernel_initializer='normal', activation='relu'))
    model.add(Dense(40, kernel_initializer='normal', activation='relu'))
    model.add(Dense(1, kernel_initializer='normal', activation='sigmoid'))
    # Compile model
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    model.fit(train_x, train_y, epochs=epochs, verbose=1, batch_size=batch_size)

    return model


# fix random seed for reproducibility
seed = 7
np.random.seed(seed)

dataframe = pd.read_csv(data)
dataset = dataframe.values  # matrix like

# x is input, y is output
x = dataset[:, 2:121].astype(float)  # for each molecule/row use the feature info from col 2-119
y = dataset[:, 0].astype(int)  # todo need the cast to int???
s = dataset[:, 1] #smiles

# encode class values as integers
#encoder = LabelEncoder()
#encoder.fit(y)
#encoded_Y = encoder.transform(y)
# print(encoded_Y)

# split data to train and test model
train_x, test_x, train_y, test_y, train_s, test_s = train_test_split(x, y,s,
                                                    train_size=0.7,shuffle=False)  # 0.7 in example as standard
# normal run
# estimator.fit(train_x,train_y)
# print(estimator.predict(test_x))
estimator = KerasClassifier(build_fn=create_baseline)
epochs = [50, 100, 200, 250, 300] #-> 0.78 on test 0.97 on train with old batch
batch_size = [int(x) for x in np.linspace(start = 100, stop = len(train_x), num = 10)] # this yields 0.81 on test and 0.98 for train
#[100, 200, 300, 400, len(train_x)]
param_grid = dict(epochs=epochs, batch_size=batch_size)
# GridSearch cannot run in parallel
grid = GridSearchCV(estimator=estimator, param_grid=param_grid, n_jobs=1, scoring='roc_auc',verbose=1,cv=5)
grid.fit(train_x, train_y)

test_res = grid.predict(test_x)
train_res = grid.predict(train_x)

pred_test = []
pred_train = []
for i in train_res:
    pred_train.append(i[0])

for i in test_res:
    pred_test.append(i[0])

############################# plots

fpr, tpr, _ = metrics.roc_curve(train_y, grid.predict_proba(train_x)[:,1])

auc_score = metrics.auc(fpr, tpr)

plt.plot(fpr, tpr, label='AUC = {:.2f}'.format(auc_score), color='red')
plt.title('ROC Curve')
plt.plot([0, 1], [0, 1], color='grey', linestyle=':')

plt.show()

###################################

print("Train Accuracy :: ", accuracy_score(train_y, train_res))
print("Test Accuracy  :: ", accuracy_score(test_y, test_res))
print("Confusion matrix test", confusion_matrix(test_y, pred_test)) # tn, fp, fn, tp
print("Confusion matrix train", confusion_matrix(train_y, pred_train)) # tn, fp, fn, tp
print("MCC  :: ", matthews_corrcoef(test_y, test_res))
print("MCC  :: ", matthews_corrcoef(train_y, train_res))


#write outFile to evaluate
with open(outfile, 'w') as out:
    #out.write("SMILES,Bio-activity,Predicted bio-activity \n")
    # write test data
    for line in range(0, len(pred_train)):
        out.write(str(train_s[line]) + "," + str(list(train_y)[line]) + "," + str(pred_train[line]) + "\n")
    # write train data
    for line in range(0, len(pred_test)):
        out.write(str(test_s[line]) + "," + str(list(test_y)[line]) + "," + str(pred_test[line]) + "\n")

