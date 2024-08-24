import pandas as pd
import sklearn
from sklearn.neighbors import KNeighborsClassifier
from sklearn import preprocessing
import sklearn.model_selection
import pickle


data=pd.read_csv('car-data.csv',sep=',')
le=preprocessing.LabelEncoder()
buying=le.fit_transform(list(data['buying']))
maint=le.fit_transform(list(data['maint']))
persons=le.fit_transform(list(data['persons']))
safety=le.fit_transform(list(data['safety']))
classcars=le.fit_transform(list(data['class']))
door = le.fit_transform(list(data ["door"]))
predict='class'
x=list(zip(buying,maint,persons,safety))
y=list(classcars)
x_train,x_test,y_train,y_test=sklearn.model_selection.train_test_split(x,y,test_size=0.1)
best=0
model=KNeighborsClassifier(n_neighbors=9)
model.fit(x_train,y_train)
acc=model.score(x_test,y_test)
if acc>best:
    best=acc
    print(best)
    with open ("carSafety.pickle",'wb') as f:
        pickle.dump(model,f)
pickleread=open("carSafety.pickle",'rb')
model=pickle.load(pickleread)
prediction=model.predict(x_test)
names=['unacc','acc','good','vgood']
for i in range(len(y_test)):
    print("Predicted = ",names[prediction[i]],"actual = ",names[y_test[i]])

