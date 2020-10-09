import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score,precision_score,recall_score
import numpy as np
from sklearn.tree import export_graphviz
import graphviz
from IPython.display import Image

df = pd.read_csv('https://sololearn.com/uploads/files/titanic.csv')
df['male']=df['Sex']=='male'

'''x=df[['Pclass','male','Age','Siblings/Spouses','Parents/Children','Fare']].values
y=df['Survived'].values

kf = KFold(n_splits=5,shuffle=True,random_state=10)

for criterion in ['gini','entropy']:
    print("Decision Tree -{}".format(criterion))
    accuracy=[]
    precision=[]
    recall=[]
    for train_index, test_index in kf.split(x):
        x_train,x_test,y_train,y_test=x[train_index],x[test_index],y[train_index],y[test_index]
        dt = DecisionTreeClassifier(criterion=criterion)
        dt.fit(x_train, y_train)
        y_pred=dt.predict(x_test)
        accuracy.append(accuracy_score(y_test,y_pred))
        precision.append(precision_score(y_test,y_pred))
        recall.append(recall_score(y_test,y_pred))
    print("accuracy:", np.mean(accuracy))
    print("precision:",np.mean(precision))
    print("recall:",np.mean(recall))'''
        
        
feature_names=['Pclass','male']
x=df[feature_names].values
y=df['Survived'].values

dt = DecisionTreeClassifier()
dt.fit(x,y)
dot_file=export_graphviz(dt,feature_names=feature_names)
graph = graphviz.Source(dot_file)
graph.render(filename='tree',format='png',cleanup=True)

