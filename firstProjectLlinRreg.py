import pandas as pd
import numpy as np
import sklearn
from sklearn import linear_model
import sklearn.model_selection
import matplotlib.pyplot as pyplot
import pickle
from matplotlib import style
data =pd.read_csv("student-mat.csv",sep=';')
data=data[['G1','G2','G3','studytime','failures','absences']]
predict='G3'
x = data.drop(columns=[predict])
y=np.array(data[predict])
x_train,x_test,y_train,y_test=sklearn.model_selection.train_test_split(x,y,test_size=0.1)
'''best=0
for _ in range(3000000):
 x_train,x_test,y_train,y_test=sklearn.model_selection.train_test_split(x,y,test_size=0.1)
 linear=linear_model.LinearRegression()
 linear.fit(x_train,y_train)
 acc=linear.score(x_test,y_test)
 
 if acc>best:
    best=acc
    print(best)
    with open('studentmodel.pickle','wb') as f:
        pickle.dump(linear,f)'''
pickle_in=open('studentmodel.pickle','rb')
linear=pickle.load(pickle_in)
predictions=linear.predict(x_test)
# for i in range(len(predictions)):
#     print(predictions[i],y_test[i])
p='G1'
style.use('ggplot')
pyplot.scatter(data[p],data['G3'])
pyplot.xlabel(p)
pyplot.ylabel('Marks in final(predicted)')
pyplot.show()