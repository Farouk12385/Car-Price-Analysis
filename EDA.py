                                                ###Import Library
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.impute        import SimpleImputer, KNNImputer
from sklearn.preprocessing import (OrdinalEncoder, OneHotEncoder,
                                   StandardScaler, MinMaxScaler,
                                   RobustScaler, PowerTransformer)
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split
from scipy.ndimage           import gaussian_filter1d
from scipy.stats             import zscore

"""
                                             we use lab 2 

"""

                                            ###Load DataSet
df = pd.read_csv('car_price.csv')
print(f'Shape: {df.shape}')       


                                            ###How many rows and columns does the dataset have?
df.head()



                                            ###Which features are numerical? Which are categorical?
num_cols = df.select_dtypes(include=['int64', 'float64']).columns
cat_cols = df.select_dtypes(include=['object']).columns

print("Numerical:", num_cols)
print("Categorical:", cat_cols)




                                            ###Are there any missing values? How many, and in which columns?
print(df.isnull().sum())




                                            ###What does the distribution of car prices look like? 
df[["price"]].hist(figsize=(8, 5), bins=20, edgecolor='black')

plt.suptitle('Distribution of Car Prices', y=0.99)
plt.tight_layout()
plt.show()
     





                                            ###Which features seem most related to price?  
corr = df.select_dtypes(include=np.number).corr()

plt.figure(figsize=(13, 10))
sns.heatmap(corr, annot=True, fmt='.2f', cmap='coolwarm',
            center=0, linewidths=0.4, annot_kws={'size': 7})
plt.title('Pearson Correlation Matrix')
plt.tight_layout()
plt.show()







