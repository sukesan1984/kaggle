import sys
sys.path.append('.')
import pandas as pd
import numpy as np
import keras
from keras.models import Seuential

result = pd.read_csv('preprocessed.csv')

# Dataを学習データと検証データに分ける
y = result.Survived.astype(int).values
X = result = result.drop('PassengerId', axis=1).drop('Survived', axis=1).fillna(0).values

(X_train, X_test, y_train, y_test) = train_test_split(
        X,y, test_size=0.3, random_state=0,
        )

model = Sequential()
model.add(Dense(12, input_dim=111, init='uniform', activation='relu'))
model.add(Dense(8, init='uniform', activation='relu'))
model.add(Dense(1, init='uniform', activation='sigmoid'))

# compile model
model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

# モデル学習
model.fit(X_train, y_train, nb_epoch=150, batch_size=10)

# モデル評価
scores = model.evaluate(X_train, y_train)


print("%s: %.2f%%" % (model.metrics_names[1], scores[1] * 100))


def predict(x, y, model):
    size = len(x)
    correct_num = 0
    for i in range(size):
        #print(x[i])
        print(model.predict(x[i]))
        # predictの結果で値がでかい方のindex
        value = np.round(model.predict(x[i]))
        if(value == y[i]):
            correct_num += 1
    print(size)
    print(correct_num)
    print("正解率:")
    print(correct_num / size)

predict(X_tsest, y_test, model)
