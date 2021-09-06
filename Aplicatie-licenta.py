
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
import seaborn as sn
import matplotlib.pyplot as plt

def citire(string):
    dataR =dataR = pd.read_csv(string)
    return dataR

def procesare(dataM):
    dataM['duration']=dataM['duration'].replace('-',0.0)
    dataM['orig_bytes']=dataM['orig_bytes'].replace('-',0)
    dataM['resp_bytes']=dataM['resp_bytes'].replace('-',0)
    dataM['duration'] = dataM['duration'].astype('float')
    dataM['orig_bytes'] = dataM['orig_bytes'].astype('int64')
    dataM['resp_bytes'] = dataM['resp_bytes'].astype('int64')
    for d in dataM:
        if dataM[d].dtypes =="object":
            dataM[d] = dataM[d].astype('category')
            dataM[d] = dataM[d].cat.codes
    dataM['label']= dataM['label'].replace(2,1)
    dataM['label']= dataM['label'].replace(3,1)
    return dataM
#locatie date
strain='C:\\Users\\Bogdan\\Desktop\\Licenta\\trainData.csv'
stest='C:\\Users\\Bogdan\\Desktop\\Licenta\\testData.csv'

#citire date
data = citire(strain)
dataT =citire(stest)
#inlocuim toate - cu 0
data = data.replace('-',0)
dataT = dataT.replace('-',0)
#creare unei copi 
dataC = data.copy()
dataCT = dataT.copy()
#procesare - modificare date
dataC = procesare(dataC);
dataCT = procesare(dataCT);         

#procesare - separare etichete de caracteristici

X = dataC[['id.resp_h','duration']]
y = dataC['label']

XT = dataCT[['id.resp_h','duration']]
yT = dataCT['label']

#X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.25,random_state=0)

X_train = X
X_test=XT
y_train=y
y_test=yT


estimatori = input("Numarul de estimatori:\n")
estimatori=int(estimatori)
adancime = input("Adancime arbore:\n")
adancime=int(adancime)
valoare_aleatoare=input("Numarul de aleator:\n")
valoare_aleatoare=int(valoare_aleatoare)
min_imp=input("Numarul minim de impartitri:\n")
min_imp=int(min_imp)
#creare si invatare model
clf = RandomForestClassifier(n_estimators=estimatori,max_depth=adancime, random_state=valoare_aleatoare,bootstrap=(False),min_samples_split=min_imp)

#clf = RandomForestClassifier(n_estimators=10,max_depth=5, random_state=12,bootstrap=(False),min_samples_split=85)

clf.fit(X_train,y_train)
#validare - prezicere
y_pred=clf.predict(X_test)

#validare -verificare rezultate 
confusion_matrix = pd.crosstab(y_test, y_pred, rownames=['Actual'], colnames=['Predicted'])
sn.heatmap(confusion_matrix, annot=True)
#print ('Accuracy: ',clf.score(X_test, y_test)*100)
print('Accuracy: ',metrics.accuracy_score(y_test, y_pred)*100,'%' )
plt.show()