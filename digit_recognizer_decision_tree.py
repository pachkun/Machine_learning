from sklearn import tree
import csv

data_train = []
date_answer = []
data_test = []

with open('./resource/train.csv') as csvfile:
    data = list(csv.reader(csvfile))
    data.pop(0)
    zipped = list(zip(*data))
    date_answer = zipped[0]
    zipped.pop(0)
    data_train = list(zip(*zipped))

clf = tree.DecisionTreeClassifier()
clf = clf.fit(data_train, date_answer)

with open('./resource/test.csv') as csvfile:
    data_test = list(csv.reader(csvfile))
    data_test.pop(0)

rec = [['ImageId', 'Label']]

for num, inp in enumerate(clf.predict(data_test)):
    rec.append([num+1, inp])

with open('./resource/res_decision_tree.csv', 'w', newline='') as csvfile:
    file = csv.writer(csvfile, delimiter=',')
    file.writerows(rec)