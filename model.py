
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.neural_network import MLPRegressor
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline


data =pd.read_csv('data.csv')
data.drop('User_ID', axis=1, inplace=True)


X = data.drop(columns='Calories')
y = data['Calories']


cat_var = [var for var in data.columns if data[var].dtypes == 'object']


class Scaling(BaseEstimator, TransformerMixin):
    def __init__(self, cat_var):
        self.cat_var = cat_var
        self.scaler = StandardScaler()
        self.value_count_mapping = {}

    def fit(self, X, y=None):
        X_new = X.copy()
        for cols in self.cat_var:
            self.value_count_mapping[cols] = X[cols].value_counts().to_dict()
        for cols in self.cat_var:
            X_new[cols] = X_new[cols].map(self.value_count_mapping[cols]).fillna(0)   
        self.scaler.fit(X_new)
        return self
    
    def transform(self, X, y=None):
        X_transformed = X.copy()
        for cols in self.cat_var:
            X_transformed[cols] = X_transformed[cols].map(self.value_count_mapping[cols]).fillna(0)
        return self.scaler.transform(X_transformed)


pipeline = Pipeline([
    ('scaler', Scaling(cat_var)),
    ('regrr', MLPRegressor(random_state=1, max_iter=1000))
])


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.35, random_state=42)



pipeline.fit(X_train, y_train)


y_pred = pipeline.predict(X_test)


print(r2_score(y_test, y_pred))
print(mean_absolute_error(y_test, y_pred))
print(mean_squared_error(y_test, y_pred))


import pickle
with open('model.pkl', 'wb') as f:
          pickle.dump(pipeline, f)





