import sys
sys.path.append('.')
import pandas as pd
import numpy as np
import re
from sklearn.model_selection import train_test_split
from common.two_layer_net import TwoLayerNet
from common.original_net import OriginalNet
from common.optimizer import SGD
import matplotlib.pyplot as plt

result = pd.read_csv('preprocessed.csv')

# Dataを学習データと検証データに分ける
X = result.drop('PassengerId', axis=1).drop('Survived', axis=1).fillna(0).values
n_examples = len(X)
y = result['Survived'].values.reshape(n_examples, -1).astype(np.int32)


(X_train, X_test, y_train, y_test) = train_test_split(
        X,y, test_size=0.3, random_state=0,
        )

print(X_train.shape)
print(y_train.shape)
print(X_test.shape)
print(y_test.shape)

# ハイパーパラメータ
input_size = 11
max_epoch = 300
batch_size = 30
hidden_size = 5
hidden_sizes = [6, 3, 6, 4, 6, 3, 6, 4, 3]
learning_rate = 0.1

# model
#model = TwoLayerNet(input_size=input_size, hidden_size=hidden_size, output_size=2)
model = OriginalNet(input_size=input_size, hidden_sizes=hidden_sizes, output_size=2)
#model.show()
optimizer = SGD(lr=learning_rate)

data_size = len(X_train)
max_iters = data_size // batch_size
total_loss = 0
loss_count = 0
loss_list = []

for epoch in range(max_epoch):
    idx = np.random.permutation(data_size)
    x = X_train[idx]
    t = y_train[idx]

    for iters in range(max_iters):
        batch_x = x[iters*batch_size:(iters+1)*batch_size]
        batch_t = t[iters*batch_size:(iters+1)*batch_size]

        loss = model.forward(batch_x, batch_t)
        model.backward()
        optimizer.update(model.params, model.grads)

        total_loss += loss
        loss_count += 1

        if (iters+1) % 10 == 0:
            avg_loss = total_loss / loss_count
            print('| epoch %d | iter %d / %d | loss %.2f'
                    % (epoch + 1, iters + 1, max_iters, avg_loss))
            loss_list.append(avg_loss)
            total_loss, loss_count = 0, 0

res_model = OriginalNet(input_size=input_size, hidden_sizes=hidden_sizes, output_size=2)
res_model.load_params('train.pyc')
#res_model.show()

def predict(x, y, model):
    size = len(x)
    correct_num = 0
    for i in range(size):
        #print(x[i])
        print(model.predict(x[i]))
        # predictの結果で値がでかい方のindex
        value = np.argmax(model.predict(x[i]))
        if(value == y[i]):
            correct_num += 1
    print(size)
    print(correct_num)
    print("正解率:")
    print(correct_num / size)

predict(X_test, y_test, res_model)
#predict(X_test, y_test, model)

