import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.tree import DecisionTreeClassifier
import matplotlib.pyplot as plt



cancer_data = load_breast_cancer()
df = pd.DataFrame(cancer_data['data'], columns = cancer_data['feature_names'])
df['target']=cancer_data['target']


x=df[cancer_data.feature_names].values
y=df['target'].values


#x_train,x_test,y_train,y_test = train_test_split(x,y,random_state=101)

rf= RandomForestClassifier(n_estimators=10, random_state=111)
#rf.fit(x_train,y_train)

worst_cols = [col for col in df.columns if 'worst' in col]

x_worst=df[worst_cols]
x_train,x_test,y_train,y_test=train_test_split(x_worst,y,random_state=101)
rf.fit(x_train,y_train)
print(rf.score(x_test,y_test))

#ft_imp=pd.Series(rf.feature_importances_,index=cancer_data.feature_names).sort_values(ascending=False)
#print(ft_imp.head(10))

#first_row = x_test[0]

#print('prediction:', rf.predict([first_row]))
#print('true value', y_test[0])
#print('random forest accuracy:', rf.score(x_test,y_test))

#dt = DecisionTreeClassifier()
#dt.fit(x_train,y_train)
#print('decision tree accuracy:', dt.score(x_test,y_test))

'''n_estimators = list(range(1,101))

param_grid={
    'n_estimators':n_estimators,
    }
gs=GridSearchCV(rf,param_grid,scoring='f1',cv=5)
gs.fit(x,y)

scores = gs.cv_results_['mean_test_score']
plt.plot(n_estimators,scores)
plt.xlabel('n_estimators')
plt.ylabel('accuracy')
plt.xlim(0,100)
plt.ylim(0.9,1)
plt.show()'''



