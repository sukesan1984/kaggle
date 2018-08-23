import sys
sys.path.append('.')
import pandas as pd
import numpy as np
import re

from sklearn.model_selection import train_test_split
#from common.two_layer_net import TwoLayerNet
from common.original_net import OriginalNet
from common.optimizer import SGD
import matplotlib.pyplot as plt

train = pd.read_csv('data/train.csv') #前処理
#性別を0, 1に
train['Sex'] = train['Sex'].str.replace('female', '1').replace('male', '0')
train.Sex = train.Sex.astype(int)

#平均値で埋める
train['Age'].fillna(train.Age.mean(), inplace=True)

#Ticketの分類
def separate(remain, regexp, result):
    # 抽出する
    df = remain[remain.Ticket.str.match(regexp)]
    # 残り
    remain = remain[~remain.Ticket.str.match(regexp)]

    df = df.join(df.Ticket.str.extract(regexp, expand=True), how='inner')
    if result is None:
        result = df
    else:
        result = pd.concat([result, df], axis=0)
    #train["TicketLetter"] = df.TicketLetter
    #train["TicketNumber"] = df.TicketNumber
    return result, remain

result,remain = separate(train, '(?P<TicketLetter>.*)\s(?P<TicketNumber>\d*)', None)
result,remain = separate(remain, '^(?P<TicketLetter>)(?P<TicketNumber>\d*)$', result)
result,remain = separate(remain, '(?P<TicketLetter>(?:C\.A)|(?:CA)|(?:C\.A\.))(?P<TicketNumber>\d*)', result)
result,remain = separate(remain, '^(?P<TicketLetter>LINE)(?P<TicketNumber>)$', result)

def replace_ticket_letter(result, regexp, category):
    result.loc[result.TicketLetter.str.match(regexp, na=False), 'TicketLetter'] = category
    return result

result = replace_ticket_letter(result, '^$', '0')
result = replace_ticket_letter(result, '^(C\.A)|(CA)|(C\.A\.)\s$', '1')
result = replace_ticket_letter(result, '^S\.P\.$', '2')
result = replace_ticket_letter(result, '^(A/5)|(A\.5)|(A\./5\.)|(A\.5\.)$', '3')
result = replace_ticket_letter(result, '^PC$', '4')
result = replace_ticket_letter(result, '^PP$', '5')
result = replace_ticket_letter(result, '^(S\.W\.)|(SW)|(SW/PP)\s$', '6')
result = replace_ticket_letter(result, '^P/PP$', '7')
result = replace_ticket_letter(result, '^A/S$', '8')
result = replace_ticket_letter(result, '^STON.*$', '9')
result = replace_ticket_letter(result, '^SC/P.*$', '10')
result = replace_ticket_letter(result, '^SC/AH.*$', '11')
result = replace_ticket_letter(result, '^SOTON.*$', '12')
result = replace_ticket_letter(result, '^S\.O\.C\.$', '13')
result = replace_ticket_letter(result, '^S\.C\./A\.4\.$', '14')
result = replace_ticket_letter(result, '^S\.C\./PARIS$', '15')
result = replace_ticket_letter(result, '^S\.O\./P.P\.$', '16')
result = replace_ticket_letter(result, '^C$', '17')
result = replace_ticket_letter(result, '^(W\./C.)|(W/C)$', '18')
result = replace_ticket_letter(result, '^LINE$', '19')
result = replace_ticket_letter(result, '^(F\.C\.\s)|(F\.C\.)|(F\.C\.C\.)$', '20')
result = replace_ticket_letter(result, '^SCO/W$', '21')
result = replace_ticket_letter(result, '^(WE/P)|(W\.E\.P\.)$', '22')
result = replace_ticket_letter(result, '^(A/4)|A4\.$', '23')
result = replace_ticket_letter(result, '^Fa$', '24')
result = replace_ticket_letter(result, '^SO/C$', '25')
result = replace_ticket_letter(result, '^SC$', '26')
result = replace_ticket_letter(result, '^S\.O\.P\.$', '27')

result.loc[result.TicketNumber.str.match('^$', na=False), 'TicketNumber'] = 0
result.TicketNumber = result.TicketNumber.astype(int)

result.TicketLetter = result.TicketLetter.astype(int)
#print(result.sort_values('TicketLetter').TicketLetter.value_counts())
result = result.drop('Ticket', axis=1)






# Nameの分類
def replace_name(result, regexp, category):
    #print(result[result.Name.str.match(regexp)])
    result.loc[result.Name.str.match(regexp), 'Name'] = category
    return result

#print(result[
#    ~result.Name.str.match('.*((Mr\.)|(Mr)).*')
#    &~result.Name.str.match('.*((Mrs\.)|(Mrs)).*')
#    &~result.Name.str.match('.*((Miss\.)|(Ms)|(Ms\.)).*')
#    &~result.Name.str.match('.*Master\..*')
#    &~result.Name.str.match('.*Dr\..*')
#    &~result.Name.str.match('.*Rev\..*')
#    &~result.Name.str.match('.*Major\..*')
#    &~result.Name.str.match('.*Col\..*')
#    ])
result = replace_name(result, '.*((Mrs\.)|(Mrs)).*', '1')
result = replace_name(result, '.*((Mr\.)|(Mr)).*', '2')
result = replace_name(result, '.*((Miss\.)|(Ms)|(Ms\.)).*', '3')
result = replace_name(result, '.*Master\..*', '4')
result = replace_name(result, '.*Dr\..*', '5')
result = replace_name(result, '.*Rev\..*', '6')
result = replace_name(result, '.*Major\..*', '7')
result = replace_name(result, '.*Col\..*', '8')
result = replace_name(result, '\D+', '9') # 最後に残ったのを変換
result.Name = result.Name.astype(int)

#Embarked変換
result.Embarked = result.Embarked.replace('S', 1).replace('C', 2).replace('Q', 3).fillna(4)

#Cabin変換
#A-F
#数字
#作戦 A-Fを1-8にして一桁目にEncodingする
#複数席あるやつは、全部数字にして足し合わせる？
def replace_cabin(result):
    result.Cabin = result.Cabin.fillna(0)
    #print(result.Cabin.unique())
    #print(result[result.Cabin.str.extract('(\w\d{0,3})')].unique())
    result.Cabin = result.Cabin.replace('(?:A(\d{0,3}))', r"\1x1", regex=True)
    result.Cabin = result.Cabin.replace('(?:B(\d{0,3}))', r"\1x2", regex=True)
    result.Cabin = result.Cabin.replace('(?:C(\d{0,3}))', r"\1x3", regex=True)
    result.Cabin = result.Cabin.replace('(?:D(\d{0,3}))', r"\1x4", regex=True)
    result.Cabin = result.Cabin.replace('(?:E(\d{0,3}))', r"\1x5", regex=True)
    result.Cabin = result.Cabin.replace('(?:F(\d{0,3}))', r"\1x6", regex=True)
    result.Cabin = result.Cabin.replace('(?:G(\d{0,3}))', r"\1x7", regex=True)
    result.Cabin = result.Cabin.replace('(?:T(\d{0,3}))', r"\1x8", regex=True)
    #単純に複数のやつは結合してみる
    result.Cabin = result.Cabin.replace('(\d*)\s(\d)', r'\1x0\2', regex=True)
    result.Cabin = result.Cabin.replace('(\d*)x(\d)', r'\1\2', regex=True)
    result.Cabin = result.Cabin.astype(int)
    print(result.Cabin.unique())

replace_cabin(result)

result = result.astype(float)

print(result.info(10))

# Dataを学習データと検証データに分ける
y = result.Survived.astype(int).values
X = result.drop('PassengerId', axis=1).drop('Survived', axis=1).values

print(X[~np.isnan(X)])

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
#hidden_size = 5
hidden_sizes = [6, 3, 6, 4, 6, 3, 6, 4, 3]
learning_rate = 0.1

# model
#ymodel = TwoLayerNet(input_size=12, hidden_size=hidden_size, output_size=2)
model = OriginalNet(input_size=input_size, hidden_sizes=hidden_sizes, output_size=2)
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
    model.save_params('train.pyc')


res_model = OriginalNet(input_size=input_size, hidden_sizes=hidden_sizes, output_size=2)
res_model.load_params('train.pyc')

def predict(x, y, model):
    size = len(x)
    correct_num = 0
    for i in range(size):
        # predictの結果で値がでかい方のindex
        value = np.argmax(model.predict(x[i]))
        if(value == y[i]):
            correct_num += 1
    print(size)
    print(correct_num)
    print("正解率:")
    print(correct_num / size)

predict(X_test, y_test, res_model)

