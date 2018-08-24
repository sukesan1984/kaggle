from chainer import Chain, optimizers, Variable
import chainer.functions as F
import chainer.links as L

import pandas as pd
import numpy as np

from sklearn import datasets, model_selection

import seaborn as sns
import matplotlib.pyplot as plt

class MLP(Chain):
    """ 4層のニューラルネットワーク(MLP)
    """
    def __init__(self, n_hid1=100, n_hid2=100, n_out=10):
        # Chainer.Chainクラスを敬称して、Chainクラスの機能を使うためにsuper関数を使う
        super().__init__()

        with self.init_scope():
            self.l1 = L.Linear(None, n_hid1)
            self.l2 = L.Linear(n_hid1, n_hid2)
            self.l3 = L.Linear(n_hid2, n_out)
    def __call__(self, x):
        hid = F.relu(self.l1(x))
        hid2 = F.relu(self.l2(hid))
        return self.l3(hid2)

result = pd.read_csv('preprocessed.csv')

# Dataを学習データと検証データに分ける
#y = result.Survived.astype(int).values
X = result.drop('PassengerId', axis=1).drop('Survived', axis=1).drop('TicketNumber', axis=1).drop('TicketLetter', axis=1).drop('Name', axis=1).fillna(0).values
n_examples = len(X)
y = result['Survived'].values.reshape(n_examples, -1).astype(np.int32)

(train_data, test_data, train_label, test_label) = model_selection.train_test_split(
        X,y, test_size=0.3, random_state=0,
        )


model = MLP(10, 10, 2)

optimizer = optimizers.SGD()

optimizer.setup(model)

train_data_variable = Variable(train_data.astype(np.float32))
train_label_variable = Variable(train_label.astype(np.int32))

loss_log = []
for epoch in range(200):
    model.cleargrads()

    prod_label = model(train_data_variable)
    loss = F.softmax_cross_entropy(prod_label, train_label_variable)
    loss.backward()
    optimizer.update()
    loss_log.append(loss.data)

#print(loss_log)

print(test_data[0:10])
print("-----")
test_data_variable = Variable(test_data.astype(np.float32))
y = model(test_data_variable)

y = F.softmax(y)
print(y.data[0:50])
pred_label = np.argmax(y.data, 1)
print(pred_label[0:50])
print("----")
print(test_label[0:50])

acc = np.sum(pred_label == test_label) / len(test_label)
print(acc)


