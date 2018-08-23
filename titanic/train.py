import sys
sys.path.append('.')
import pandas as pd
import numpy as np
import re
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

# Dataを入力と出力に分ける

