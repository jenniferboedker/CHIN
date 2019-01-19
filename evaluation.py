# script to evaluate the predictors classification
# by Jennifer Bödker, Tobias Nietsch

import argparse
import sys
import math


# compare the
def parse_supervised(filename):
    true = []
    pred = []

    for line in open(filename):
        if "Predict" in line:
            continue
        line = line.split(",")
        true.append(int(line[1]))
        pred.append(int(line[2]))

    return true,pred

def parse_unsupervised(filename):
    smiles=[]
    true = []
    pred1 = []
    pred2 = []

    for line in open(filename):
        if "Predict" in line:
            continue
        line = line.split(",")
        smiles.append(line[0])
        true.append(int(line[1]))
        if line[2].strip() == 'cluster_1':
            pred2.append(1)
            pred1.append(-1)
        else:
            pred1.append(1)
            pred2.append(-1)

    tpr1, fpr1,mcc = rates(true,pred1)
    tpr2, fpr2,mcc = rates(true,pred2)

    if tpr1 > tpr2:
        with open(filename, 'w') as out:
            out.write("SMILES,Bio-activity,Predicted bio-activity \n")
            # write test data
            [out.write(str(i) + "," +str(j) + "," + str(z)+"\n") for i, j, z in zip(smiles, true, pred1)]

        return true,pred1
    else:
        with open(filename, 'w') as out:
            out.write("SMILES,Bio-activity,Predicted bio-activity \n")
            # write test data
            [out.write(str(i) + "," +str(j) + "," + str(z)+"\n") for i, j, z in zip(smiles, true, pred2)]
        return true,pred2



def rates(true=list,pred=list):

    TP = 0
    FP = 0
    FN = 0
    TN = 0

    for i in range(0,len(true)):
        if true[i] == 1 and pred[i] == 1:
            TP += 1
        elif true[i] == 1 and pred[i] == -1:
            FN += 1
        elif true[i] == -1 and pred[i] == -1:
            TN += 1
        elif true[i] == -1 and pred[i] == 1:
            FP += 1

    tpr = float(TP/(TP+FN))
    fpr = float(FP/(FP+TN))
    #print(TP,TN,FP,FN)
    #print(fpr,tpr)
    mcc = float((TP*TN - FP*FN)/(math.sqrt((TP+FP)*(TP+FN)*(TN+FP)*(TN+FN))))


    return tpr,fpr, mcc



def main(argv):
    # Parameter handeling
    parser = argparse.ArgumentParser(description='Evaluation of Predictors')
    # mandatory input parameters
    parser.add_argument('-i', type=str, required=True, help='path to input file (comma separated csv; true,predicted)')
    parser.add_argument('-t', type=str, required=True, help='result of which predictor? (supervised, unsupervised)')
    parser.add_argument('-o', type=str, required=True, help='path to output file')

    args = parser.parse_args()

    input = args.i
    type = args.t
    #outputFile = args.o

    # Parse the input file to generate a list of RDKit molecules
    if type == 'supervised':
        true,pred = parse_supervised(input)
    else:
        true,pred = parse_unsupervised(input)


    #calculate percentage of right predicted values
    right = 0
    for t in range(0,len(true)):
        if true[t] == pred[t]:
            right += 1

    print(right/len(true))

    # calc TPR and FPR
    tpr,fpr,mcc = rates(true,pred)
    print("tpr: "+str(tpr)+" fpr: "+str(fpr)+" mcc: "+str(mcc))


if __name__ == "__main__":
    main(sys.argv[1:])