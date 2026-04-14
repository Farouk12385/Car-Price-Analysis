 # Task 5: KNN Classification
#======================================== 
   
q1 = train['price'].quantile(0.34)
q2 = train['price'].quantile(0.66)

def categorize_price(price):
    if price <= q1:
        return 'Cheap'
    elif price <= q2:
        return 'Moderate'
    else:
        return 'Expensive'

train['price_category'] = train['price'].apply(categorize_price)
test['price_category']  = test['price'].apply(categorize_price)

print("Class Distribution (Train):")
print(train['price_category'].value_counts())

X_train = train.drop(['price', 'price_category'], axis=1)
y_train = train['price_category']

X_test = test.drop(['price', 'price_category'], axis=1)
y_test = test['price_category']
 
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV

knn = KNeighborsClassifier()

param_grid = {
    'n_neighbors': [3, 5, 7, 9],
    'metric': ['euclidean', 'manhattan']
}

grid_search = GridSearchCV(
    estimator=knn,
    param_grid=param_grid,
    cv=5,
    scoring='accuracy',
    n_jobs=-1
)

grid_search.fit(X_train, y_train)

print("\nBest Parameters:", grid_search.best_params_)
print("Best Cross Validation Score:", grid_search.best_score_)

from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

best_model = grid_search.best_estimator_

y_pred = best_model.predict(X_test)

print("\nTest Accuracy:", accuracy_score(y_test, y_pred))

print("\nClassification Report:\n")
print(classification_report(y_test, y_pred))

import seaborn as sns
import matplotlib.pyplot as plt

cm = confusion_matrix(y_test, y_pred)

plt.figure()
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=best_model.classes_,
            yticklabels=best_model.classes_)

plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()                                           
