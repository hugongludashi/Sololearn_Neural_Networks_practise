from sklearn.model_selection import KFold
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score,recall_score,f1_score
import pandas as pd
import numpy as np

df = pd.read_csv('https://sololearn.com/uploads/files/titanic.csv')
df['male'] = df['Sex']=='male'

kf=KFold(n_splits=5, shuffle=True)

x1=df[['Pclass','male','Age','Siblings/Spouses','Parents/Children','Fare']].values
x2 = df[['Pclass','male','Age']].values
x3=df[['Fare','Age']].values
y=df['Survived'].values


def score_model(x,y,kf):
    accuracy_scores=[]
    precision_scores=[]
    recall_scores=[]
    f1_scores=[]
    splits=kf.split(x)
    for train_index, test_index in splits:
        x_train=x[train_index]
        y_train=y[train_index]
        x_test = x[test_index]
        y_test = y[test_index]
        model =LogisticRegression()
        model.fit(x_train,y_train)
        y_pred=model.predict(x_test)
        accuracy_scores.append(accuracy_score(y_test,y_pred))
        precision_scores.append(precision_score(y_test,y_pred))
        recall_scores.append(recall_score(y_test,y_pred))
        f1_scores.append(f1_score(y_test,y_pred))
    print('accuracy:',np.mean(accuracy_scores))
    print('precision:',np.mean(precision_scores))
    print('recall:',np.mean(recall_scores))
    print('f1 score:',np.mean(f1_scores))
    
print("Logistic Regression with all features")
score_model(x1,y,kf)
print()
print("Logistic Regression with Pclass, Sex & Age features")
print(score_model(x2,y,kf))
print()
print('Logistic Regression with Fare, Age')
print(score_model(x3,y,kf))
