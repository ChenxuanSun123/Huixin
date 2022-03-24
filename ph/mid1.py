import pandas as pd
import re

'''Import training data and testing data'''

train_data = pd.read_csv('train_data.csv')
test_data = pd.read_csv('test_data.csv')

train_data.describe()
train_data.info()
'''We can see that no data is missing,
    there are about 22% of loan default'''

train_data['Employment.Type'].value_counts()
train_data['Employment.Type'].value_counts()
train_data['loan_default'].value_counts()

from datetime import datetime, date

def age(born):
    '''We transform date of birth into age'''
    born = datetime.strptime(born, '%d/%m/%Y').date()
    today = date.today()
    return today.year - born.year - ((today.month, today.day) < (born.month, born.day))
train_data['Age'] = train_data['Date.of.Birth'].apply(age)


missing = train_data.isnull().sum()
'''There are 5679 null in employment type, which is about 3%, 
    I used the backward fill method.
'''
train_data['Employment.Type'] = train_data['Employment.Type'].fillna(method = 'bfill')

'''We transform average account age and credit history length into months'''
train_data['AVERAGE.ACCT.AGE'] = train_data['AVERAGE.ACCT.AGE'].map(lambda x: re.sub("[^0-9]+"," ",x))
train_data['AVERAGE.ACCT.AGE'] = train_data['AVERAGE.ACCT.AGE'].str.split(" ",expand=True)[0].astype(int)*12+train_data['AVERAGE.ACCT.AGE'].str.split(" ",expand=True)[1].astype(int)
train_data['CREDIT.HISTORY.LENGTH'] = train_data['CREDIT.HISTORY.LENGTH'].map(lambda x: re.sub("[^0-9]+"," ",x))
train_data['CREDIT.HISTORY.LENGTH'] = train_data['CREDIT.HISTORY.LENGTH'].str.split(" ",expand=True)[0].astype(int)*12+train_data['CREDIT.HISTORY.LENGTH'].str.split(" ",expand=True)[1].astype(int)


'''building the pipeline for the numerical attributes'''
from sklearn.base import BaseEstimator, TransformerMixin
class DataFrameSelector(BaseEstimator, TransformerMixin):
    def __init__(self, attribute_names):
        self.attribute_names = attribute_names
    def fit(self, X, y=None):
        return self
    def transform(self, X):
        return X[self.attribute_names]
    
sel = DataFrameSelector(['Age','disbursed_amount', 'asset_cost', 'ltv', 'PRI.NO.OF.ACCTS','PRI.ACTIVE.ACCTS', 'PRI.OVERDUE.ACCTS',
                         'PRI.CURRENT.BALANCE','PRI.SANCTIONED.AMOUNT','PRI.DISBURSED.AMOUNT','AVERAGE.ACCT.AGE','CREDIT.HISTORY.LENGTH',
                         'PRIMARY.INSTAL.AMT','NEW.ACCTS.IN.LAST.SIX.MONTHS','DELINQUENT.ACCTS.IN.LAST.SIX.MONTHS',
                         'NO.OF_INQUIRIES'])
sel.transform(train_data)
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

num_pipeline = Pipeline([
        ("select_numeric", DataFrameSelector(['Age','disbursed_amount', 'asset_cost', 'ltv', 'PRI.NO.OF.ACCTS','PRI.ACTIVE.ACCTS', 'PRI.OVERDUE.ACCTS',
                         'PRI.CURRENT.BALANCE','PRI.SANCTIONED.AMOUNT','PRI.DISBURSED.AMOUNT','AVERAGE.ACCT.AGE','CREDIT.HISTORY.LENGTH',
                         'PRIMARY.INSTAL.AMT','NEW.ACCTS.IN.LAST.SIX.MONTHS','DELINQUENT.ACCTS.IN.LAST.SIX.MONTHS',
                         'NO.OF_INQUIRIES'])),
        ("imputer", SimpleImputer(strategy="median")),
    ])
num_pipeline.fit_transform(train_data)


'''building the pipeline for the categorical attributes'''
class MostFrequentImputer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        self.most_frequent_ = pd.Series([X[c].value_counts().index[0] for c in X],
                                        index=X.columns)
        return self
    def transform(self, X, y=None):
        return X.fillna(self.most_frequent_)

from sklearn.preprocessing import OneHotEncoder

'''We aggregate all similar perform cns score description together
'''
print(train_data['PERFORM_CNS.SCORE.DESCRIPTION'].value_counts())
train_data = train_data.replace({'PERFORM_CNS.SCORE.DESCRIPTION':{'No Bureau History Available' : 'No score',
                                                                  'C-Very Low Risk' : 'Low', 'A-Very Low Risk' : 'Low',
                                                                  'D-Very Low Risk' : 'Low', 'B-Very Low Risk' : 'Low',
                                                                  'M-Very High Risk' : 'High', 'F-Low Risk' : 'Low',
                                                                  'K-High Risk':'High','H-Medium Risk':'Medium','E-Low Risk':'Low',
                                                                  'I-Medium Risk':'Medium','G-Low Risk':'Low',
                                                                  'Not Scored: Sufficient History Not Available' :'No score',
                                                                  'J-High Risk':'High', 'Not Scored: Not Enough Info available on the customer':'No score',
                                                                  'Not Scored: No Activity seen on the customer (Inactive)':'No score',
                                                                  'Not Scored: No Updates available in last 36 months':'No score',
                                                                  'L-Very High Risk':'High', 'Not Scored: Only a Guarantor':'No score',
                                                                  'Not Scored: More than 50 active Accounts found':'No score'}})
print(train_data['PERFORM_CNS.SCORE.DESCRIPTION'].value_counts())

cat_pipeline = Pipeline([
        ("select_cat", DataFrameSelector(["Employment.Type", "MobileNo_Avl_Flag", "Aadhar_flag",
                                          'PAN_flag', 'VoterID_flag','Driving_flag','Passport_flag'
                                          ])),
        ("imputer", MostFrequentImputer()),
        ("cat_encoder", OneHotEncoder(sparse=False)),
    ])

'''We join the category pipeline and numerical pipeline toghter'''

from sklearn.pipeline import FeatureUnion
preprocess_pipeline = FeatureUnion(transformer_list=[
        ("num_pipeline", num_pipeline),
        ("cat_pipeline", cat_pipeline),
    ])


X_train = preprocess_pipeline.fit_transform(train_data)
y_train = train_data['loan_default']

from matplotlib import pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
import numpy as np

'''Split our training data into validation set and training set,
we use 80% of data as training set and 20% as validation set.'''
train_X, vali_X, train_y, vali_y = train_test_split(X_train, y_train, train_size=0.8)

'''Logistic regression method'''
scaler = StandardScaler()
lr = LogisticRegression()
lr_pipe = Pipeline([('standardize', scaler),
                   ('lr', lr)])

lr_params = {
    'lr__C': [1, 10, 100, 500, 1000]
}

lr_search = GridSearchCV(lr_pipe, lr_params, cv=5, scoring='roc_auc' ,n_jobs=-1)

lr_search.fit(train_X, train_y)
print('Best parameters are:\n', lr_search.best_params_)
print('AUC is ', lr_search.best_score_)

'''KNN method
'''
scaler = StandardScaler()
knn = KNeighborsClassifier()
knn_pipe = Pipeline(steps = [
    ('standardize', scaler),
    ('knn', knn)
])

knn_params = {
    'knn__n_neighbors':  np.arange(0,10,2),
    'knn__weights':      ['uniform', 'distance'],
    'knn__p':            [1, 2],
}

knn_search = GridSearchCV(knn_pipe, knn_params, cv=5, scoring='roc_auc' ,n_jobs=-1)
knn_search.fit(train_X, train_y)

print('Best parameters are:\n', knn_search.best_params_)
print('AUC is ', knn_search.best_score_)






def search_threshold(y_score, y_true):
    list_the = []
    list_acc = []

    for i in np.arange(0,1,0.01):
        list_the.append(i)
        y_pred = np.where(y_score>=i, 1, 0)
        acc = np.mean(y_pred == y_true)
        list_acc.append(acc)
    best_the = np.array(list_the)[list_acc == max(list_acc)]
    return best_the[0].item()

lr_prob = lr_search.predict_proba(vali_X)[:, 1]
lr_the = search_threshold(lr_prob, vali_y)
lr_pred = np.where(lr_prob>=lr_the, 1, 0)


def measurement(y_score, y_pred, y_true):
    fpr, tpr, thresholds = roc_curve(y_true, y_score, pos_label=1)
    plt.plot(fpr, tpr)
    plt.show()
    vali_accuracy = accuracy_score(y_true, y_pred)*100
    vali_auc_roc = roc_auc_score(y_true, y_score)*100
    print('Confusion matrix:\n', confusion_matrix(y_true, y_pred))
    print('Validation accuracy: %.4f %%' % vali_accuracy)
    print('Validation AUC: %.4f %%' % vali_auc_roc)

print('The logistic regression')
measurement(lr_prob, lr_pred, vali_y)



'''Do the same thing to test data'''
test_data['Age'] = test_data['Date.of.Birth'].apply(age)
test_data['Employment.Type'] = test_data['Employment.Type'].fillna(method = 'bfill')
test_data['AVERAGE.ACCT.AGE'] = test_data['AVERAGE.ACCT.AGE'].map(lambda x: re.sub("[^0-9]+"," ",x))
test_data['AVERAGE.ACCT.AGE'] = test_data['AVERAGE.ACCT.AGE'].str.split(" ",expand=True)[0].astype(int)*12+test_data['AVERAGE.ACCT.AGE'].str.split(" ",expand=True)[1].astype(int)
test_data['CREDIT.HISTORY.LENGTH'] = test_data['CREDIT.HISTORY.LENGTH'].map(lambda x: re.sub("[^0-9]+"," ",x))
test_data['CREDIT.HISTORY.LENGTH'] = test_data['CREDIT.HISTORY.LENGTH'].str.split(" ",expand=True)[0].astype(int)*12+test_data['CREDIT.HISTORY.LENGTH'].str.split(" ",expand=True)[1].astype(int)
test_data = test_data.replace({'PERFORM_CNS.SCORE.DESCRIPTION':{'No Bureau History Available' : 'No score',
                                                                  'C-Very Low Risk' : 'Low', 'A-Very Low Risk' : 'Low',
                                                                  'D-Very Low Risk' : 'Low', 'B-Very Low Risk' : 'Low',
                                                                  'M-Very High Risk' : 'High', 'F-Low Risk' : 'Low',
                                                                  'K-High Risk':'High','H-Medium Risk':'Medium','E-Low Risk':'Low',
                                                                  'I-Medium Risk':'Medium','G-Low Risk':'Low',
                                                                  'Not Scored: Sufficient History Not Available' :'No score',
                                                                  'J-High Risk':'High', 'Not Scored: Not Enough Info available on the customer':'No score',
                                                                  'Not Scored: No Activity seen on the customer (Inactive)':'No score',
                                                                  'Not Scored: No Updates available in last 36 months':'No score',
                                                                  'L-Very High Risk':'High', 'Not Scored: Only a Guarantor':'No score'}})

X_test = preprocess_pipeline.fit_transform(test_data)
pred = lr_search.predict(X_test)
pred = pd.DataFrame(pred, columns=['pred'])
#pred.to_csv('Prediction.csv', index=0)
from sklearn.metrics import classification_report
print(classification_report(vali_y,lr_pred))








