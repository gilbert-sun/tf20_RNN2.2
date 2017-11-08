import csv
import glob
import numpy as np
files = glob.glob("./*.csv")
for fileNameInput in files:

        fp = open(fileNameInput, "r").readlines()
        print("\n\n--------------------------------------Begin")
        print(fileNameInput)
        print ("--------------------------------------")
        cfp = csv.DictReader(fp)
        dataset1 = list(cfp)
        N = len(dataset1)
        M = int(N / 20)
        L = 20 * (M-1) + 3
        print ("N:" + str(N)+"  M(N/20):" +str(M)+"  L[ 20 * (M-1) + 3]:"+ str(L))
        print ("--------------------------------------\n\n")
        dataset2 = dataset1[:L]
        data_label = []
        data_features = []
        # print ("--------------------------------------"1111111111111)
        # print (M)
        # print ("--------------------------------------")
        for i in range(L):
            # print(str(i) + ":" + "\n")
            try:
                w = float(dataset2[i]["Voltage"].strip('"'))
                w1 = float(dataset2[i - 1]["Voltage"].strip('"'))
                w3 = abs(w1 - w)
                if (w3 > 0.4):
                    # print("Level4")
                    data_label.append(4.)
                    data_features.append(w3)
                elif (w3 < 0.4 and w3 > 0.3):
                    # print("Level3")
                    data_label.append(3.)
                    data_features.append(w3)
                elif (w3 < 0.3 and w3 > 0.2):
                    # print("Level2")
                    data_label.append(2.)
                    data_features.append(w3)
                elif (w3 < 0.2):
                    # print("Level1")
                    data_label.append(1.)
                    data_features.append(w3)
                    # if (i < 20):
                    #     print(data_features )
                    #     print(data_label )
            except ValueError:
                print("Error with row", i, ":", dataset2[i])
                pass

        data_reshape =  np.reshape(data_features,(M-1,20,1))

        label_reshape = np.reshape(data_label,(M-1,20,1))

        print ("--------------------------------------")
        print ("--------------------------------------End")
        #print (data_reshape)
        print (label_reshape)
        #print ( data_features )
        # print ("--------------------------------------")
        # print ("--------------------------------------")