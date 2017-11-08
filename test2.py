import csv
import glob
import numpy as np
files = glob.glob("./*.csv")
for fileNameInput in files:

    fp = open(fileNameInput, "r").readlines()
    print("\n\n--------------" + fileNameInput)
    print("--------------------------------------\n\n")
    cfp = csv.DictReader(fp)
    dataset1 = list(cfp)
    N = len(dataset1)
    print("--------------------------------------")
    print("N:" + str(N) )
    print("--------------------------------------")
    data_label = []
    data_features = []
    for i in range(N):
        print(str(i) + ":" + "\n")
        try:
            w = float(dataset1[i]["Voltage"].strip('"'))
            w1 = float(dataset1[i - 1]["Voltage"].strip('"'))
            w3 = abs(w1 - w)
            if (w3 > 0.4):
                print("Level4")
                data_label.append(4)
                data_features.append(w3)
            elif (w3 < 0.4 and w3 > 0.3):
                print("Level3")
                data_label.append(3)
                data_features.append(w3)
            elif (w3 < 0.3 and w3 > 0.2):
                print("Level2")
                data_label.append(2)
                data_features.append(w3)
            elif (w3 < 0.2):
                # print("Level1")
                data_label.append(1)
                data_features.append(w3)


        except ValueError:
            print("Error with row", i, ":", dataset1[i])
            pass