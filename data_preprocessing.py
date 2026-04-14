                                               ###Import Library
from EDA import df
                                                ###Import Library
import numpy as np
import pandas as pd



from sklearn.impute        import SimpleImputer
from sklearn.preprocessing import OneHotEncoder,MinMaxScaler 
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split



"""
                         we use lab 2 also 
    

"""


#  Drop rows with no target value (price=NaN) — cannot train on them
#  Also drop rows where ALL values are NaN

target = 'price'
df.dropna(subset=[target], inplace=True)
df.dropna(how='all', inplace=True)
print(f'Cleaned    — Shape after dropping missing target rows: {df.shape}')





                                   ###Train/Test to prevent leakage
train, test = train_test_split(
    df, test_size=0.2, random_state=1   
)
print(f'Train: {train.shape}  |  Test: {test.shape}')

# Define features and target — from train/test only, no leakage
X_train = train.drop(columns='price')
y_train  = train['price']

X_test  = test.drop(columns='price')
y_test  = test['price']

print(f'Features: {X_train.shape[1]}  |  Train samples: {X_train.shape[0]}  |  Test samples: {X_test.shape[0]}')





                                    ###Handle missing values — drop rows, fill with mean/mode
                                    
# Numerical columns: impute with median (robust to outliers) 

numeric_cols = train.select_dtypes(include=np.number).columns.tolist()
numeric_imputer = SimpleImputer(strategy='median')

train[numeric_cols] = numeric_imputer.fit_transform(train[numeric_cols]) 
test[numeric_cols] = numeric_imputer.transform(test[numeric_cols])

# Categorical column: impute with mode
cat_cols = train.select_dtypes(include=['object', 'category']).columns.tolist()
categorical_imputer = SimpleImputer(strategy='most_frequent')

train[cat_cols] = categorical_imputer.fit_transform(train[cat_cols])
test[cat_cols] = categorical_imputer.transform(test[cat_cols])

# Confirm
print('Remaining missing values in train:', train.isnull().sum().sum())
print('Remaining missing values in test:', test.isnull().sum().sum())






                                       ###Encode categorical columns — use  One-Hot Encoding. because is no ordinal relationship
cat_cols = train.select_dtypes(include=['object', 'category']).columns


enc = OneHotEncoder(handle_unknown="ignore", sparse_output=False)


train_enc = enc.fit_transform(train[cat_cols])
test_enc  = enc.transform(test[cat_cols])


ohe_cols = enc.get_feature_names_out(cat_cols)

train_ohe = pd.DataFrame(train_enc, columns=ohe_cols, index=train.index)
test_ohe  = pd.DataFrame(test_enc,  columns=ohe_cols, index=test.index)

train = train.drop(cat_cols, axis=1)
test  = test.drop(cat_cols, axis=1)

train = pd.concat([train, train_ohe], axis=1)
test  = pd.concat([test,  test_ohe],  axis=1)



print(f'Encoding    Train: {train.shape} | Test: {test.shape} | OHE cols added: {len(ohe_cols)}')




                                 ###Detect and handle outliers — use  IQR method
outlier_cols = numeric_cols 
bounds = {}
 
for col in outlier_cols:
    Q1  = train[col].quantile(0.25)
    Q3  = train[col].quantile(0.75)
    IQR = Q3 - Q1
    bounds[col] = (Q1 - 1.5 * IQR, Q3 + 1.5 * IQR)
 
total_capped_train = sum(
    ((train[col] < lo) | (train[col] > hi)).sum()
    for col, (lo, hi) in bounds.items()
)
 
for col, (lower, upper) in bounds.items():
    train[col] = train[col].clip(lower, upper)
    test[col]  = test[col].clip(lower, upper)
 
print(f'Outliers  Capped {total_capped_train} values in train across {len(outlier_cols)} columns (IQR method)')







                                ###Scale numerical features — important for KNN. Use  MinMaxScaler. 

num_cols = train.select_dtypes(include=np.number).columns
scaler = MinMaxScaler()

train[num_cols] = scaler.fit_transform(train[num_cols])
test[num_cols]  = scaler.transform(test[num_cols])
print(f'Scaling    {len(num_cols)} features scaled to [0,1] | min={train[num_cols].min().min():.2f} | max={train[num_cols].max().max():.2f}')
 
 
 
 
 
 
 



                                           ### Export data

train.to_csv('train_preprocessed.csv', index=False)
test.to_csv('test_preprocessed.csv',   index=False)
 
print(f'Exported   train_preprocessed.csv ({train.shape[0]} rows) & test_preprocessed.csv ({test.shape[0]} rows) saved.')
                                           