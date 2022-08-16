


import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
import matplotlib.pyplot as plt
import pickle




# import os
# for dirname, _, filenames in os.walk('/kaggle/input'):
#     for filename in filenames:
#         print(os.path.join(dirname, filename))
        
        
        
data = pd.read_csv('Training_Table_Long.csv')

#data.drop(['Lstd', 'Hn', 'Hnmax', 'hstd', 'hb', 'T1', 'Deff', 'MaxDr', 'IndexStructureNo'], axis=1)

data.head()

data.isnull().sum()

#Get Target data 
y = data['ADE_frame_disp_m']

#Load X Variables into a Pandas Dataframe with columns 
X = data.drop(['ADE_frame_disp_m'], axis = 1)

print(f'X : {X.shape}')

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=101)

print(f'X_train : {X_train.shape}')
print(f'y_train : {y_train.shape}')
print(f'X_test : {X_test.shape}')
print(f'y_test : {y_test.shape}')



#Fitting 3 folds for each of 96 candidates, totalling 288 fits
#{'bootstrap': True,
# 'max_depth': 4,
# 'max_features': 'auto',
# 'min_samples_leaf': 1,
# 'min_samples_split': 5,
# 'n_estimators': 80}


# Number of trees in random forest
#n_estimators = [int(x) for x in np.linspace(start = 10, stop = 80, num = 10)]
#n_estimators = 80
n_estimators = 200
# Number of features to consider at every split
#max_features = ['auto', 'sqrt']
max_features = 'auto'
# Maximum number of levels in tree
#max_depth = [2,4]
max_depth=20
# Minimum number of samples required to split a node
#min_samples_split = [2, 5]
min_samples_split = 10
# Minimum number of samples required at each leaf node
#min_samples_leaf = [1, 2]
#min_samples_leaf = 1
min_samples_leaf = 8
# Method of selecting samples for training each tree
#bootstrap = [True, False]
bootstrap = True
# 4 8 15 150



rf_Model = RandomForestRegressor(n_estimators=n_estimators, max_features=max_features, max_depth=max_depth, min_samples_split=min_samples_split, min_samples_leaf=min_samples_leaf, bootstrap=(bootstrap))



rf_Model.fit(X_train, y_train)
print('Model is fit')

# save the model to disk
filename = 'finalized_model.sav'
pickle.dump(rf_Model, open(filename, 'wb'))



test_data=pd.read_csv('Test_Table_Long.csv')
real_values_col=test_data['ADE_frame_disp_m']
T=test_data.drop('ADE_frame_disp_m', axis=1)


predictions=rf_Model.predict(T)


time=np.linspace(0,118/2,119)

pp=990
plt.plot(time, predictions[pp*118:(pp+1)*118+1],'r')
plt.plot(time, real_values_col[pp*118:(pp+1)*118+1],'b')
names=test_data['IndexStructureNo']
print(names[pp*118:(pp+1)*118+1])
plt.xlabel('Time (sec)')
plt.ylabel('Frame Horizontal Displacement at the Effective Height (m)')
plt.legend(['Prediction', 'Analytically Determined'])
plt.grid()
plt.savefig('990.png')


np.savetxt('Predictions.csv', predictions, delimiter=",")
np.savetxt('Real_Values.csv', real_values_col, delimiter=",")


#print (f'Train Accuracy - : {rf_Grid.score(X_train,y_train):.3f}')
#print (f'Test Accuracy - : {rf_Grid.score(X_test,y_test):.3f}')






