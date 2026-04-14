import pandas as pd

train = pd.read_csv('train_preprocessed.csv')
test  = pd.read_csv('test_preprocessed.csv')

# ── Regression targets ──────────────────────────────────────
y_train_reg = train['price']
y_test_reg  = test['price']

# ── Classification targets ──────────────────────────────────
# Thresholds from TRAIN only — no leakage
cheap_threshold     = train['price'].quantile(0.33)
expensive_threshold = train['price'].quantile(0.66)

def classify_price(price):
    if price <= cheap_threshold:
        return 'Cheap'
    elif price <= expensive_threshold:
        return 'Moderate'
    else:
        return 'Expensive'

train['price_category'] = train['price'].apply(classify_price)
test['price_category']  = test['price'].apply(classify_price)  # same thresholds

y_train_clf = train['price_category']
y_test_clf  = test['price_category']

# ── Stats ───────────────────────────────────────────────────
print(f"Cheap threshold:     {cheap_threshold:.4f}")
print(f"Expensive threshold: {expensive_threshold:.4f}")
print("\nTrain distribution:")
print(y_train_clf.value_counts())
print("\nTest distribution:")
print(y_test_clf.value_counts())
# ============================================