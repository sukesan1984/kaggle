import pandas as pd
import numpy as np
import re
train = pd.read_csv('data/train.csv') #前処理
#性別を0, 1に
train['Sex'] = train['Sex'].str.replace('female', '1').replace('male', '0')

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

result.TicketLetter = result.TicketLetter.astype(int)
#print(result.sort_values('TicketLetter').TicketLetter.value_counts())
result = result.drop('Ticket', axis=1)
print(result.head(10))


# Nameの分類
def replace_name(result, regexp, category):
    result.loc[result.Name.str.match(regexp), 'Name'] = category

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
result = replace_name(result, '.*((Mr\.)|(Mr)).*', 1)
