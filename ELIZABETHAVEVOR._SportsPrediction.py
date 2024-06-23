#!/usr/bin/env python
# coding: utf-8

# In[1]:


#import the necessary libraries
import pandas as pd


# In[2]:


import numpy as np


# In[3]:


from sklearn.compose import ColumnTransformer, make_column_selector


# In[4]:


from sklearn.impute import SimpleImputer


# In[5]:


from sklearn.pipeline import Pipeline


# In[6]:


from sklearn.preprocessing import StandardScaler, OneHotEncoder


# In[7]:


from sklearn.model_selection import train_test_split, cross_val_score


# In[8]:


from sklearn.linear_model import LinearRegression


# In[9]:


from sklearn.metrics import mean_squared_error


# In[10]:


#load the given datasets
#dataset 1 for training the model
males_legacy = pd.read_csv("C:\\Users\\user\\OneDrive - Ashesi University\\intro to ai\\data\\players_21.csv", na_values = '-')
males_legacy


# In[11]:


males_legacy.info()


# In[12]:


males_legacy.describe()


# In[13]:


print(males_legacy.columns)


# In[14]:


males_legacy.shape


# In[15]:


# Drop unnecessary columns
columns_to_drop = ['nation_logo_url','club_flag_url', 'club_logo_url','player_url', 'player_face_url', 'nation_flag_url']  
males_legacy.drop(columns=columns_to_drop, axis=1, inplace=True)
males_legacy


# In[16]:


# males_legacy_df = pd.DataFrame(males_legacy, columns = ['overall', 'height


# In[17]:


males_legacy.iloc[100:500, :]


# In[18]:


threshold = 0.4
L = []
L_more = []
for col in males_legacy.columns:
    if males_legacy[col].isnull().sum() < threshold * len(males_legacy):
        L.append(col)
    else:
        L_more.append(col)
    


# In[19]:


males_legacy = males_legacy[L]
males_legacy


# In[20]:


#separate numeric and non-numeric features
numeric_data = males_legacy.select_dtypes(include = np.number)
non_numeric = males_legacy.select_dtypes(include = ['object'])
#multivariate imputation
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
imp = IterativeImputer(max_iter = 10, random_state = 0)
#impute data, transform, round it off and put to dataframe
numeric_data = pd.DataFrame(np.round(imp.fit_transform(numeric_data)), columns = numeric_data.columns)
numeric_data #holds all values without missing values


# In[21]:


#males_legacy['dribblin'].isnull().sum()


# In[22]:


#deal with non numeric data
non_numeric_columns = ['short_name', 'long_name', 'player_positions',
    'club_name', 'league_name', 'club_position', 
    'nationality_name', 'preferred_foot', 'work_rate','real_face']
# y = non_numeric[non_numeric_columns]
non_numeric = males_legacy[non_numeric_columns]
males_legacy = males_legacy.drop(columns = non_numeric_columns)


# In[23]:


# One-hot encode the remaining non-numeric data
non_numeric = pd.get_dummies(non_numeric, drop_first=True).astype(int)
non_numeric


# In[24]:


Xtrain = pd.concat([numeric_data, non_numeric], axis = 1)
Xtrain


# In[25]:


Xtrain.shape


# In[26]:


all_columns = numeric_data.shape[1] + non_numeric.shape[1]  
print("Expected: ", all_columns)
print("Actual: ", Xtrain.shape[1])


# In[27]:


all_columns = numeric_data.columns.tolist() + non_numeric.columns.tolist()


# In[28]:


df_Xtrain = pd.DataFrame(Xtrain, columns = all_columns)
df_Xtrain


# In[29]:


#define variables
X_train = Xtrain.drop(columns = ['overall'])
Ytrain = males_legacy['overall']


# In[30]:


#scale x
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_train = scaler.fit_transform(Xtrain)
X_train


# In[31]:


# X_test1 = Xtrain.drop(columns = ['overall'])
# Ytest1 = pmales_legacy['overall']


# In[32]:


all_columns = numeric_data.columns.tolist() + non_numeric.columns.tolist()


# In[33]:


correlations = {}
for col in all_columns:
    correlations[col] = np.corrcoef(X_train[:, all_columns.index(col)], Ytrain)[0, 1]
    


# In[34]:


#convert to dataframe
correlation_df = pd.DataFrame(list(correlations.items()), columns = ['feature', 'correlation with dependent']).sort_values(by = 'correlation with dependent', ascending = False)
correlation_df


# In[35]:


best_features = correlation_df['feature'].head(10).tolist()
best_features


# In[36]:


#select the corresponding columns from the data for training
best_X_train = X_train[:, [all_columns.index(feature) for feature in best_features]]
best_X_train


# In[37]:


from sklearn.ensemble import RandomForestRegressor
import pickle as pkl


# In[38]:


#training first model with cross validation
randomforest = RandomForestRegressor(n_estimators = 100, random_state = 42)
pkl.dump(randomforest, open("C:\\Users\\user\\OneDrive - Ashesi University\\intro to ai\\data" + randomforest.__class__.__name__ + '.pkl', 'wb'))
cv_scores = cross_val_score(randomforest, best_X_train, Ytrain, cv = 5, scoring = 'neg_mean_squared_error')
cv_rmse = np.sqrt(-cv_scores)
print("Cross validation RMSE: " , cv_rmse)
print("Mean cross validation RMSE: ", cv_rmse.mean())

#train model on the data
randomforest.fit(best_X_train, Ytrain)
# y_pred = randomforest.predict(X_test)
    # print(f"{model.__class__.__name__} RMSE: {np.sqrt(mean_squared_error(y_test, y_pred))}")
    # print(f"{model.__class__.__name__} R2: {r2_score(y_test, y_pred)}").....testing phase


# In[39]:


from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.tree import DecisionTreeRegressor


# In[40]:


#training second model
decisiontree = DecisionTreeRegressor()
pkl.dump(decisiontree, open("C:\\Users\\user\\OneDrive - Ashesi University\\intro to ai\\data" + decisiontree.__class__.__name__ + '.pkl', 'wb'))
#perform a 7-fold cross validation
cv_scores = cross_val_score(decisiontree, best_X_train, Ytrain, cv = 5, scoring = 'neg_mean_squared_error')
cv_rmse = np.sqrt(-cv_scores)
print("Cross validation RMSE: " , cv_rmse)
print("Mean cross validation RMSE: ", cv_rmse.mean())


#train model on the data
decisiontree.fit(best_X_train, Ytrain)



# In[41]:


from sklearn.linear_model import LinearRegression


# In[42]:


#training the third model
linearregression = LinearRegression()
pkl.dump(linearregression, open("C:\\Users\\user\\OneDrive - Ashesi University\\intro to ai\\data" + linearregression.__class__.__name__ + '.pkl', 'wb'))
#perform a 7-fold cross validation
cv_scores = cross_val_score(linearregression, best_X_train, Ytrain, cv = 5, scoring = 'neg_mean_squared_error')
cv_rmse = np.sqrt(-cv_scores)
print("Cross validation RMSE: " , cv_rmse)
print("Mean cross validation RMSE: ", cv_rmse.mean())


#train model on the data
linearregression.fit(best_X_train, Ytrain)


# In[ ]:





# In[43]:


from sklearn.ensemble import  GradientBoostingRegressor


# In[44]:


gradient_boosting =  GradientBoostingRegressor(n_estimators = 100, random_state = 42)
pkl.dump(gradient_boosting, open("C:\\Users\\user\\OneDrive - Ashesi University\\intro to ai\\data" + gradient_boosting.__class__.__name__ + '.pkl', 'wb'))
cv_scores = cross_val_score(gradient_boosting, best_X_train, Ytrain, cv = 5, scoring = 'neg_mean_squared_error')
cv_rmse = np.sqrt(-cv_scores)
print("Cross validation RMSE: " , cv_rmse)
print("Mean cross validation RMSE: ", cv_rmse.mean())

#train model on the data
gradient_boosting.fit(best_X_train, Ytrain)


# In[ ]:





# In[45]:


from sklearn.ensemble import VotingRegressor


# In[46]:


voting = VotingRegressor(estimators=[('lr', linearregression),('gb', gradient_boosting),('rf', randomforest), ('dt', decisiontree)])
pkl.dump(voting, open("C:\\Users\\user\\OneDrive - Ashesi University\\intro to ai\\data" + voting.__class__.__name__ + '.pkl', 'wb'))
cv_scores = cross_val_score(voting, best_X_train, Ytrain, cv = 5, scoring = 'neg_mean_squared_error')
cv_rmse = np.sqrt(-cv_scores)
print("Cross-validation RMSE: " , cv_rmse)
print("Mean cross validation RMSE: ", cv_rmse.mean())


voting.fit(best_X_train, Ytrain)


# In[ ]:





# In[47]:


from sklearn.ensemble import AdaBoostRegressor


# In[48]:


ada=AdaBoostRegressor(estimator=decisiontree,n_estimators=100)
pkl.dump(ada, open("C:\\Users\\user\\OneDrive - Ashesi University\\intro to ai\\data" + ada.__class__.__name__ + '.pkl', 'wb'))
cv_scores = cross_val_score(ada, best_X_train, Ytrain, cv = 5, scoring = 'neg_mean_squared_error')
cv_rmse = np.sqrt(-cv_scores)
print("Cross-validation RMSE: " , cv_rmse)
print("Mean cross validation RMSE: ", cv_rmse.mean())


ada.fit(best_X_train, Ytrain)


# In[ ]:





# In[49]:


#gridsearch with cross validation and tuning to choose best model
from sklearn.model_selection import GridSearchCV, KFold


# In[50]:


rf = RandomForestRegressor()

PARAMETERS_gb ={
"max_depth":[2,5,7],
"min_samples_leaf":[1,4,7],
"min_samples_split":[1,4,7],
# "learning_rate":[0.3, 0.1, 0.03],
"n_estimators":[100,300,500]}
cv=KFold(n_splits=3)
model_gs=GridSearchCV(rf,param_grid=PARAMETERS_gb,cv=cv,scoring="neg_mean_squared_error", n_jobs=-1, verbose = 2)
model_gs.fit(best_X_train, Ytrain)
pkl.dump(model_gs, open("C:\\Users\\user\\OneDrive - Ashesi University\\intro to ai\\data" + model_gs.__class__.__name__ + '.pkl', 'wb'))


# In[51]:


model_gs.best_params_


# In[52]:


model_gs.best_score_


# In[53]:


gb = GradientBoostingRegressor()

PARAMETERS_gb ={
"max_depth":[2,5,7],
"min_samples_leaf":[1,4,7],
"min_samples_split":[1,4,7],
# "learning_rate":[0.3, 0.1, 0.03],
"n_estimators":[100,300,500]}
cv=KFold(n_splits=3)
model_gs=GridSearchCV(gb,param_grid=PARAMETERS_gb,cv=cv,scoring="neg_mean_squared_error", n_jobs=-1, verbose = 2)
model_gs.fit(best_X_train, Ytrain)
pkl.dump(model_gs, open("C:\\Users\\user\\OneDrive - Ashesi University\\intro to ai\\data" + model_gs.__class__.__name__ + '.pkl', 'wb'))


# In[54]:


model_gs.best_params_


# In[55]:


model_gs.best_score_


# In[ ]:





# In[56]:


#dataset 2 for testing the model
players_22 = pd.read_csv("C:\\Users\\user\\OneDrive - Ashesi University\\intro to ai\\data\\players_22.csv", na_values = '-')
players_22


# In[57]:


players_22.info()


# In[58]:


players_22.describe()


# In[59]:


print(players_22.columns)


# In[60]:


players_22.shape


# In[61]:


# Drop unnecessary columns
columns_to_drop2 = ['nation_logo_url','club_flag_url', 'club_logo_url','player_url', 'player_face_url', 'nation_flag_url']  
players_22.drop(columns=columns_to_drop2, axis=1, inplace=True)
players_22


# In[62]:


players_22.iloc[100:500, :]


# In[63]:


threshold = 0.4
L1 = []
L1_more = []
for col in players_22.columns:
    if players_22[col].isnull().sum() < threshold * len(players_22):
        L1.append(col)
    else:
        L1_more.append(col)
    


# In[64]:


players_22= players_22[L1]
players_22


# In[65]:


#separate numeric and non-numeric features
numeric_data1 = players_22.select_dtypes(include = np.number)
non_numeric1 = players_22.select_dtypes(include = ['object'])
#multivariate imputation
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
imp = IterativeImputer(max_iter = 10, random_state = 0)
#impute data, transform, round it off and put to dataframe
numeric_data1 = pd.DataFrame(np.round(imp.fit_transform(numeric_data1)), columns = numeric_data1.columns)
numeric_data1 #holds all values without missing values


# In[66]:


#players_22['dribblin'].isnull().sum()


# In[67]:


#deal with non-numeric data
non_numeric_columns1 = ['short_name', 'long_name', 'player_positions',
    'club_name', 'league_name', 'club_position', 
    'nationality_name', 'preferred_foot', 'work_rate','real_face']
# y = non_numeric[non_numeric_columns]
non_numeric1 = players_22[non_numeric_columns1]
players_22 = players_22.drop(columns = non_numeric_columns1)


# In[68]:


# One-hot encode the remaining non-numeric data
non_numeric1 = pd.get_dummies(non_numeric1, drop_first=True).astype(int)
non_numeric1


# In[69]:


Xtest = pd.concat([numeric_data1, non_numeric1], axis = 1)
Xtest


# In[70]:


Xtest.shape


# In[71]:


all_columns1 = numeric_data1.shape[1] + non_numeric1.shape[1]  #subtract 1 for the overall column
print("Expected: ", all_columns1)
print("Actual: ", Xtest.shape[1])


# In[72]:


all_columns1 = numeric_data1.columns.tolist() + non_numeric1.columns.tolist()


# In[73]:


df_Xtest = pd.DataFrame(Xtest, columns = all_columns1)
df_Xtest


# In[74]:


#define variables
X_test = Xtest.drop(columns = ['overall'])
Ytest = players_22['overall']


# In[75]:


#scale x
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_test = scaler.fit_transform(Xtest)
X_test


# In[76]:


all_columns1 = numeric_data1.columns.tolist() + non_numeric1.columns.tolist()


# In[77]:


correlations1 = {}
for col in all_columns1:
    correlations1[col] = np.corrcoef(X_test[:, all_columns1.index(col)], Ytest)[0, 1]
    


# In[78]:


#convert to dataframe
correlation_df1 = pd.DataFrame(list(correlations1.items()), columns = ['features', 'correlation with dependent variable']).sort_values(by = 'correlation with dependent variable', ascending = False)
correlation_df1


# In[79]:


best_features1 = correlation_df1['features'].head(25).tolist()
best_features1


# In[80]:


#select the features with that show maximum correlation from the data for testing to an array
best_X_test1 = X_test[:, [all_columns1.index(feature) for feature in best_features]]
best_X_test1


# In[ ]:





# In[89]:


#testing the models
y_pred = randomforest.predict(best_X_test1)
pkl.dump(randomforest, open("C:\\Users\\user\\OneDrive - Ashesi University\\intro to ai\\data" + randomforest.__class__.__name__ + '.pkl', 'wb'))


# In[90]:


print(f"""Mean Absolute Error = {mean_absolute_error(y_pred, Ytest)},
        Mean Squared Error = {mean_squared_error(y_pred, Ytest)}
        Root Mean Squared Error = {np.sqrt(mean_squared_error(y_pred, Ytest))},
        R2 score = {r2_score(y_pred, Ytest)}"""
)


# In[ ]:





# In[91]:


y_pred = decisiontree.predict(best_X_test1)
pkl.dump(decisiontree, open("C:\\Users\\user\\OneDrive - Ashesi University\\intro to ai\\data" + decisiontree.__class__.__name__ + '.pkl', 'wb'))


# In[92]:


print(f"""Mean Absolute Error = {mean_absolute_error(y_pred, Ytest)},
        Mean Squared Error = {mean_squared_error(y_pred, Ytest)}
        Root Mean Squared Error = {np.sqrt(mean_squared_error(y_pred, Ytest))},
        R2 score = {r2_score(y_pred, Ytest)}"""
)


# In[ ]:





# In[93]:


y_pred = linearregression.predict(best_X_test1)


# In[94]:


print(f"""Mean Absolute Error = {mean_absolute_error(y_pred, Ytest)},
        Mean Squared Error = {mean_squared_error(y_pred, Ytest)}
        Root Mean Squared Error = {np.sqrt(mean_squared_error(y_pred, Ytest))},
        R2 score = {r2_score(y_pred, Ytest)}"""
)


# In[ ]:





# In[ ]:





# In[112]:


pip install streamlit


# In[107]:


# import streamlit as st
# import numpy as np
# import pickle as pkl


# In[108]:


# pkl.dump(randomforest, open("C:\\Users\\user\\OneDrive - Ashesi University\\intro to ai\\data" + randomforest.__class__.__name__ + '.pkl', 'wb'))


# In[110]:


# #function to predict the player rating using the best model
# def predict_rating(data):
#      predict = best_model.predict([data])
#      return predict[0]

# #streamlit app operations
# st.title("Prediction of the Overall Rating of Players")

# #defining the top features
# top_features = ['overall', 'movement_reactions', 'mentality_composure', 'potential', 'release_clause_eur', 'wage_eur', 'value_eur', 'power_shot_power', 'passing', 'mentality_vision']

# #taking new input from user
# player_features = []
# for f in top_features:
#     value = st.number_input(f"Kindly enter a value for {f}:", min_value = 0.0, step = 0.1)
#     player_features.append(value)

# if st.button("Predict Player Rating"):
#     player_rating = predict_rating(player_features)
#     st.write(f"The predicted rating of the player is: {player_rating:.4f}")
    


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




