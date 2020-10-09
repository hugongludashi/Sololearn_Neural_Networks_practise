from sklearn.model_selection import KFold
from sklearn.linear_model import LogisticRegression
import pandas as pd
import numpy as np

df = pd.read_csv('https://sololearn.com/uploads/files/titanic.csv')
df['male']=df['Sex']=='male'
x=df[['Pclass','male','Age','Siblings/Spouses','Parents/Children','Fare']].values
y=df['Survived'].values

kf = KFold(n_splits=5,shuffle=True)

scores=[]
splits=list(kf.split(x))
for train_indices,test_indices in splits:
    x_train=x[train_indices]
    x_test=x[test_indices]
    y_train=y[train_indices]
    y_test=y[test_indices]
    model=LogisticRegression()
    model.fit(x_train,y_train)
    scores.append(model.score(x_test,y_test))

print(scores)
print(np.mean(scores))

final_model=LogisticRegression()
final_model.fit(x,y)
