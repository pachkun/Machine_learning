import pybrain
import csv

data_test = []

with open('./resource/train.csv') as csvfile:
    data = list(csv.reader(csvfile))
    data.pop(0)


ds = pybrain.SupervisedDataSet(784, 10)

for target in data:
    ds.addSample(target[1:], target[:1])

net = pybrain.buildNetwork(784, 30, 10, bias=True)
trainer = pybrain.BackpropTrainer(net, ds, learningrate=0.001, momentum=0.99, verbose=True)

#trainer.trainEpochs(epochs=1)

with open('./resource/test.csv') as csvfile:
    data_test = list(csv.reader(csvfile))
    data_test.pop(0)

result = []
for num, out in enumerate(data_test):
    result.append([num+1, net.activate(out)])

print(result[:100])