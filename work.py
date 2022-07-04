#!/usr/bin/env python
# coding: utf-8

# In[718]:


# ------------>>>>>>>> RUN THIS CODE CELL <<<<<<<<------------
# === CELL TYPE: IMPORTS AND SETUP 

import time      # for testing use only
import os         # for testing use only
import os                           # for testing use only
import pandas as pd
import numpy as np
import sklearn
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn import preprocessing, linear_model, model_selection
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn import linear_model, metrics, preprocessing
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.metrics import r2_score, f1_score


# In[719]:


def load_csv(file_name):
    df=pd.read_csv(file_name)
    return df


# In[726]:


file_name = '.' + os.sep + 'work' + os.sep + 'yad1.csv'
df2 = load_csv(file_name)
display(df2)


# In[727]:


def count_duplicatives(df, col_name=None):
    res = df.duplicated(keep='first').sum()
    # print(df.duplicated(keep="first").count())
    if col_name:
        res = df[col_name].duplicated(keep='first').sum()

    return (res)


# In[728]:


res_dup= count_duplicatives(df2, col_name=None)
print(res_dup)


# In[729]:


def remove_duplicatives(df, col_name=None):
    res_df=df.drop_duplicates(keep='first')
    if col_name:
        res_df=df.drop_duplicates(subset=col_name,keep='first')
    return res_df


# In[730]:


df=remove_duplicatives(df2, col_name=None)


# In[731]:


display(df)


# In[644]:


def remove_corrupt_rows(df):
    res1=df.dropna()
    return res1


# In[645]:


df3=remove_corrupt_rows(df)
df3.info()
display(df3)


# In[674]:


replace_map2 = {'וילה':7,'יחידת דיור':10,'דירת סטודיו':10,'מרתף':10,'קראוון':11,'לופט':12,'דירה להחלפה':0,
                'חניה':0,'קרקע':0,'פרטר':0,'מגרש למגורים':8,'נחלה':8,'מיני פנטהאוז':9,'דירה':1,'דירת גג':2,
                'פנטהאוז':2,'דירת גן':3,'בית פרטי':4,'דו משפחתי':4,'קוטג':4,'טריפלקס':5,'דופלקס':5,'Beer sheva':0,'משק חקלאי':6}

replace_map3={'בת ים':6, 'טבריה':1,'כטבריה':1,'טבר':1, 'רמת גן':6,'רמת גן קריניצי':6, 'נס ציונה':4, 'תל אביב יפו':6,'תל אביב יפובT':6
              ,'תל אביב - יפו':6,'5טבריה':1,'תלאביב יפו':6,'תל אביב יפ':6,'תל אביב- הצפון הישן':6,'תל אביב יפוב':6,'רחובות':3,
       'חדרה':2, 'לוד':4,'גני אביב לוד':4,'ףלוד':4, 'חריש':2, 'ראשון לציון':4,'ראשון לציוןואי':4, 'נשר':2, 'באר שבע':3, 'עכו':1,
       'חולון':6,'קרית אילון, חולון':6, 'חיפה':2,'חיפ':2,'חיפה':2,'חיפה':2,'חיפה הדר':2, 'באר יעקב':4, 'פתח תקווה':4, 'אור עקיבא':2, 'אופקים':3,
       'רמת-גן':6,'Hadera':2, 'קרית ארבע':7, 'נתניה':4, 'נתיבות':3, 'Uירושלים':5,'ירושלים':5,'ירושליםU':5, 3:'Beer sheva',
       'אשקלון':3,'Myאשקלון':3, 'אשדוד':3, 'גבעתיים':6, 'עפולה':1,'עפלה עילית':1, 'רחובות החדשה':4, 'אור יהודה':6,
       'רמלה':4, 'בני ברק':6,'בני ברק , פרדס כץ':6, 'מצפה רמון':3,'Tנהריה':1,'נהריה':1,'נהריה':1, 'נהרייה':1, 'קרית ביאליק':2,
       'ופתח תקווה':4,'Petsch Tikva':4,'קרית טבעון':2, 'קרית מוצקין':2, 'קרית שמונה':1, 'קרית מלאכי':3,
       'מודיעין מכבים רעות':4, 'קרית גת':3,'קרית גת לכיש':3, 'בית שמש':5, 'רחיפה':2, 'ניו יורק':0,
       'קריית מוצקין':2, 'יבנה':4, 'טירת כרמל':2,'טירה':0,'טירת הכרמל':2, 'בנימינה':2, 'ביתר עילית':7,
       'אילת':3, 'ערד':3,'יהוד מונוסון':4, 'צפת':1,'קריית אתא':2,'ץקרית אתא':2,'ץקרית אתא5':2, 'קרית אתא':2,
       'קציר חריש':2, 'קרית חיים':2,'קרית חיים מערבית':2, 'אפרת':7, 'ירוחם':3, 'קרית אונו':6,
       'גבעת שמואל':4, 'אלעד':4, 'חולו':6, 'ראש העין':4, 'מעלות תרשיחא':1,
       'כפר יונה':4,'Rosh HaAyin':4, 'כרמיאל':1, 'רחובו':4, 'יבנאל':0, 'פסגת זאב':5,
       'מזכרת בתיה':4,'hifa':2,'נווה מונוסון':4, "בני עי''ש":0, 'פרדס חנה - כרכור':0,
       'כפר סבא':4, 'רעננה':4, 'צור יצחק':4, 'רכסים':0, 'גני תקווה':4, 'שדרות':3,
       'בת -ים':6,'³בת ים':6, 'הרצליה':6,'הרצליה נווה עמל':6, 'מעלה אדומים':5,'כפר אדומים':5, 'אזור':6, 'יירושלים':5,'ירושליםU':5,'ירושלים גילה':5,'ירושליים':5, 'ק.מלאכי':3,
       'פתח תקווה':4,'jerusalem':5,'שוהם':4, 'נהריה':1, 'מבשרת ציון':5, 'ראשון':4,
       'הרצליה פיתוח':6,'tsfad':1,'קירית ים':2, 'רמת השרון':6, 'מוצא עילית':0, 'טבאר שבע':3,
       'רמת בית שמש':5, 'יפו':6,'קיסריה':2,'באר טוביה':3, 'יקנעם עלית':1,'יקנעם':1, 'קרית ים':2, 'תל אביב':6, 'אריאל':7,
       "ג'בע מצפון לירושלים":5,'בית-דגן':4, 'יוסף טל':0, 'רמות א':0, 'קריית מלאכי':3,
       'נתניה אזורים':4, '⁹אשדוד':3, 'יהוד מונסון':4, 'תל-אביב':6, 'בתים':6, 'יהוד':4,
       'נוף הגליל':0, 'קדימה צורן':4,'יהוד-מונוסון':4, 'גבעת זאב':0, 'מגדל העמק':1,
       'אלפי מנשה':0, 'תל א אביב':6, 'ברכפלד':0, 'פ"ת':4, 'דימונה':3, 'הוד השרון':4,
       'דאלית אל כרמל':0, 'בת-ים':6, 'ביתשמש':5, 'קריית גת':3, 'תל אב':6, 'באר-שבע':3,
       'פתח תקוה':4,'יוקנעם':1,'להבים':3, 'ראשון ךציון':4, 'גדרה':4,'מטולה':1, 'ירושלם':5, 'אשדודד':3
              ,'חצור הגלילית':0, 'אורנית':0,'כפר תבור':0,'ספיר':0,'מרכז':0,'מיתר':0,'בוסתן הגליל':0,
              'עלמון':0,'צוקים':0,'רוממה':0,'מזור':0,'ברקן':0,'בית אריה עופרים':0,'אירוס':0,'לימן':0,
              'משמרות':0,"ניל''י":0,'יגל':0,'שבות רחל':0,'בחר':0,'כפר חנניה':0,'ענב':0,'נווה ימין':0,
              'רמת השר':0,'גבעת אבני':0,'יקיר':0,'כורזים':0,'נווה אילן':0,'חד נס':0,'חבר':0,'סביון':4,
        'אבן יהודה':0,'בת חפר':2,'הר גילה':0,'זכרון יעקב':2,'כפר ורדים':0,'נצרת עילית':0,'צור הדסה':0,
        'קצרין':1,"נווה אטי''ב":0,'אפיק':0,'נווה אטיב':0,'בית זית':0,'נוה חוף':0,'עומר':0,
        'כפר סאלד':0,'Beer sheva':3,'קרית עקרון':4,'תלקרית עקרון':4,'רבבה':0, 'ןאשקלון':3,'אשקלון':3,'בני יהודה':0,'רחלים':0,'אור הנר':0,
        'מזרעה':0,'כפר חיטים':0,'בית עזרא':0,'תנובות':0,'שומרה':0,'אבטליון':0,'אשלים':0,'אדמית':0,'מצפה שלם':0,
        'עין דור':0,'בורגתה':0,'קליה':0,'פורייה - נווה עובד':0,'קדומים':0,'אליכין':0,'שלומית':0,'שבי ציון':0,'עין שריד':0,'אבני חפץ':0,
        'אחוזת ברק':0,'ניר עקיבא':0,'משמר איילון':0,'תל ציון':0,'פורייה - נווה עופד':0,'אשק':0,'תפרח':0,'כפר ויתקין':0,'אבן מנחם':0,
        'באר גנים':0,'נוב':0,'פוריה נווה עוב':0,'רוחמה':0,'עלי זהב':0,'ירחיב':0,'מבוא ביתר':0,'מרכז שפירא':0,'צור משה':0,'קלנסווה':0,
        'נווה אפק הצבאית':0,'קרית היובל':0,'וולינגרד , בולגריה':0,'עתלית':0,'הולנד':0,'גילון':0,'כרכום':0,'באקה אלגרביה':0,'יונתן':0,
        'מצפה יריחו':0,'בולגריה':0,'הר אדר':0,'שערי תקווה':0,'מושב שורש':0,"תלמי ביל''ו":0,'גבעת יואב':0,'ניצן':0,'ישובי השומרון':0,
        'בארותיים':0,'הולנדU':0,'שושנת העמקים':0,'ישע':0,'שריגים':0,'הדר יוסף':0,'בר גיורא':0,'גבעתי':0,'רמת פולג':0,'כפר חסידים':0,
        'רמת מוצא':0,'עמנואל':0,'נחושה':0,'Болгария':0,'מתתיהו':0,'ספסופה':0,'כוכב יעקב':0,'צפון הישן':0,'בני ציון':0,
        'אשדות יעקב איחוד':0,'כפר מונש':0,'רמת צבי':0,'עלמון ענתות':0,'כפר שמאי':0,'בית חורון':0
        ,'שלומי':0,'Essaouira':0,'מועצה איזורית':0,'אביחיל':0,'מושב שער אפרים':0,'עראבה':0
        ,'מירון':0,'גן יבנה':4,'מדגל העמק':1,'בית אל':0,'תל תאומים':0,'בלינסון':0,'אבן שמואל':0,
        'אלון שבות':0,'מושב כרמל':0,'בית אריה':0,'נווה זוהר':0,'בית אלעזרי':0,'פוריה נווה עובד':0,
        'רמת ישי':0,'גבע בנימין (אדם)':0,'כפר מעש':0,'תל מונד':0,'אלון':0,'גיתה':0,'שני ליבנה':0,'בארותיים':0,'מעין ברוך':0,'יובלים':0,
              'בארותיים':0,'משמר הירדן':0,'קלע -רמת הגולן':1,'בארותיים':0,'שיכוני המיזרח':0,'נצרת':0,
              'חומת שמואל':0,'Tel Aviv':6,'פרדס חנה':0,'בארותיים':0,'Tel aviv':6,'בית חשמונאי':4,'רמת חן':0,
              'ראש פינה':0,'בארותיים':0,'קריית שמואל':2,'קרית שמואל חיפה':2,'רמות רמז':0,'פרדס חנה-כרכור':0,
              'דורות':0,'שדה צבי':0,'reפתח תקווה':4,'גני מודיעין':4,'מודיעין עילית':4,'כפר גנים':0,'איתמר':0,
              'גבעת עדה':0,'פרדסיה':0,'נווה חוף':0,'מורן':0,'קו הים':0,'הודיה':0,'נעלה':0,'עפרה':0,'כוכב יאיר':4,'חירבת חורה':0,
        'קרני שומרון':7,'בית שאן':1,'חשמונאים':7,'Herzilya':6, 'קריית ים':2}

#1-North dist
#2-Haifa dist
#3-Tel aviv dist
#4-Merkaz dist
#5-Jeruz dist
#6-South dist

df5=df3.copy()
df5.replace(replace_map2,inplace=True)
df5.replace(replace_map3,inplace=True)
df5.drop(index=df5[df5['type'] =='Beer sheva'].index,inplace=True)
df5.drop(index=df5[df5['room'] =='Beer sheva'].index,inplace=True)

print(df5['city'].unique())
#df13=df5['city'].unique()
#df5=df5[df5['floor']>1]
#df5=df5[df5['type']==1]
df5["price"]=df5["price"].str.replace('$','')
df5["price"]=df5["price"].str.replace('₪','')
df5["price"]=df5["price"].str.replace(' ','')
df5["price"]=df5["price"].str.replace(',','')
df5["floor"]=pd.to_numeric(df5.floor)
df5["room"]=pd.to_numeric(df5.room)
df5["price"]=pd.to_numeric(df5.price)
df5["city"]=pd.to_numeric(df5.city)
df5["type"]=pd.to_numeric(df5.type)
df5.drop(index=df5[df5['city'] ==0].index,inplace=True)
df5=df5[df5['floor']>2]
df5=df5[df5['floor']<25]
df5.drop(index=df5[df5['price'] <500000].index,inplace=True)
df5.drop(index=df5[df5['price'] >4000000].index,inplace=True)
display(df5)
df5.info()
df5.plot.scatter('city','price',figsize=(12,8),title='the connection between price to city')



# In[675]:


file_name = '.' + os.sep + 'work' + os.sep + 'yad2.csv'


# In[676]:


def load_dataset(df, label_column):
    y=df[label_column]
    x=df.drop(label_column,axis=1)
    return x,y


# In[677]:


category_col_name = 'price'
X, y = load_dataset(df5, category_col_name)
X2=X[['type','room','floor','city']]                       
display(X2)
#display(df6)
display(df5)


# In[678]:


def split_to_train_and_test(X, y, test_ratio, rand_state):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_ratio, random_state=rand_state)
    return X_train, X_test, y_train, y_test


# In[711]:


#X_train, X_test, y_train, y_test = split_to_train_and_test(X2, y, 0.3,40)#94 80 27-3.88 50-0.41 30- 4.18 40-4.188 40-4.47
for i in range(100):
    X_train, X_test, y_train, y_test = split_to_train_and_test(X2, y, 0.3,i)
    scaler=preprocessing.StandardScaler()
    #print(X_train)
    X_train_scaled=scaler.fit_transform(X_train)
    X_train_scaled=pd.DataFrame(X_train_scaled,columns=X_train.columns,index=X_train.index)
    X_test_scaled=pd.DataFrame(scaler.transform(X_test),columns=X_test.columns,index=X_test.index)
    target_col_name = 'price'  
    X_1st_train, y_1st_train = X_train_scaled, y_train
    X_1st_test, y_1st_test = X_test_scaled, y_test
    trained_model_1st = train_1st_model(X_1st_train, y_1st_train)
    pred_1st_vals = predict_1st(trained_model_1st, X_1st_test)
    y_pred_1st= pd.Series(pred_1st_vals,index=X_1st_test.index)
    eval_res_1st = evaluate_performance_1st(y_1st_test, y_pred_1st)
    #print(i)
    print(i,eval_res_1st)
#display(X_train, X_test, y_train, y_test)
#display(X2)
#df5.describe()

sns.pairplot(df5)
#sns.pairplot(df6)

#print(df5['price'])


# In[696]:


scaler=preprocessing.StandardScaler()
print(X_train)
X_train_scaled=scaler.fit_transform(X_train)
X_train_scaled=pd.DataFrame(X_train_scaled,columns=X_train.columns,index=X_train.index)
X_test_scaled=pd.DataFrame(scaler.transform(X_test),columns=X_test.columns,index=X_test.index)
#print(X_train_scaled,X_test_scaled)


# In[697]:


cor_df=df5
cor=cor_df.corr(method='pearson')
print(cor)


# In[698]:


def train_1st_model(X_train, y_train):
    m=linear_model.LinearRegression().fit(X_train,y_train)
    return m


# In[699]:


def predict_1st(trained_1st_model, X_test):
    return trained_1st_model.predict(X_test)


# In[700]:


def evaluate_performance_1st(y_test,y_predicted):
    return metrics.r2_score(y_test,y_predicted)


# In[712]:


#file_name_train = '.' + os.sep + 'data' + os.sep + 'imdb_for_1st_model_train.csv'
#file_name_test = '.' + os.sep + 'data' + os.sep + 'imdb_for_1st_model_test.csv'
target_col_name = 'price'  
X_1st_train, y_1st_train = X_train_scaled, y_train
X_1st_test, y_1st_test = X_test_scaled, y_test
trained_model_1st = train_1st_model(X_1st_train, y_1st_train)
pred_1st_vals = predict_1st(trained_model_1st, X_1st_test)
y_pred_1st= pd.Series(pred_1st_vals,index=X_1st_test.index)
eval_res_1st = evaluate_performance_1st(y_1st_test, y_pred_1st)
print(y_1st_test)
print(trained_model_1st)
print(eval_res_1st)
print(y_pred_1st)

#df5.plot.scatter('city','price',figsize=(12,8),title='the connection between price to floor')
df6=df5.copy()
df6=df6[df6['floor']>2]
df6=df6[df6['floor']<25]
df6=df6[(df6['city']==3) | (df6['city']==4)]

print(df6.city.value_counts())
display(df6)
#df5.describe(include='all')
category_col_name = 'price'
X, y = load_dataset(df6, category_col_name)
X2=X[['type','room','floor','city']]                       
#display(X2)
#display(df5)
#display(df6)
X_train, X_test, y_train, y_test = split_to_train_and_test(X2, y, 0.3,45)

scaler=preprocessing.StandardScaler()
#print(X_train)
X_train_scaled=scaler.fit_transform(X_train)
X_train_scaled=pd.DataFrame(X_train_scaled,columns=X_train.columns,index=X_train.index)
X_test_scaled=pd.DataFrame(scaler.transform(X_test),columns=X_test.columns,index=X_test.index)
X_1st_train, y_1st_train = X_train_scaled, y_train
X_1st_test, y_1st_test = X_test_scaled, y_test
trained_model_1st = train_1st_model(X_1st_train, y_1st_train)
pred_1st_vals = predict_1st(trained_model_1st, X_1st_test)
y_pred_1st= pd.Series(pred_1st_vals,index=X_1st_test.index)
eval_res_1st = evaluate_performance_1st(y_1st_test, y_pred_1st)
sns.pairplot(df6)

print(y_1st_test)
print(trained_model_1st)
print(eval_res_1st)
print(y_pred_1st)

#print(metrics.mean_absolute_error(y_1st_test,y_pred_1st))
#print(metrics.mean_squared_error(y_1st_test,y_pred_1st))
#print(df5.city.value_counts())
#df5.city.value_counts().plot(kind='pie')

#plt.scatter(y_test,y_pred_1st)
#sns.displot((y_test-y_pred_1st),bins=50)
#plt.scatter(y_test,y_pred_1st)


# In[702]:


#plt.scatter(y_test,y_pred_1st)


# In[703]:


df5.plot.scatter('price','floor',figsize=(12,8),title='the connection between price to floor')
df6.plot.scatter('price','floor',figsize=(12,8),title='the connection between price to floor')


# In[704]:


df6.plot.scatter('city','price',figsize=(12,8),title='the connection between price to floor')


# In[705]:


df6.plot.scatter('city','room',figsize=(12,8),title='the connection between price to floor')


# In[706]:


print('1-North dist\n2-Haifa dist\n3-Tel aviv dist\n4-Merkaz dist\n5-Jeruz dist\n6-South dist')


# In[707]:


df5.plot.scatter('city','floor',figsize=(12,8),title='the connection between price to floor')


# In[708]:


df5.plot.scatter('room','price',figsize=(12,8),title='the connection between price to floor')


# In[709]:


df6.plot.scatter('city','price',figsize=(12,8),title='the connection between price to floor')


# In[710]:


df5.plot.scatter('city','price',figsize=(12,8),title='the connection between price to floor')


# In[ ]:





# In[ ]:





# In[ ]:




