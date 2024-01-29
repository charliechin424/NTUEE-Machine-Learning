import csv
import pandas as pd
import numpy as np

csv_list = []
voting = []
counter = 1
prediction = np.array([], dtype=np.int32)
length = 3

for i in range(41):
    voting.append(0)

for i in range(length):
    file = open('prediction (' + str(i+1) +  ').csv')
    reader = csv.reader(file)
    data_list = list(reader) 
    csv_list.append(data_list)


for i in range(527364):
    for j in range(length):
        voting[int(csv_list[j][i+1][1])] += 1
    maxid = 0
    for k in range(41):
        if voting[k] > voting[maxid]:
            maxid = k
    prediction = np.concatenate((prediction, np.array([maxid])), axis=0)
    print(2)
    for l in range(41):
        voting[l] = 0
print(1)

with open('prediction.csv', 'w') as f:
    f.write('Id,Class\n')
    for i, y in enumerate(prediction):
        f.write('{},{}\n'.format(i, y))

file.close()