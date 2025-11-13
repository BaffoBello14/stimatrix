# 📋 Contextual Features Guidelines

**Purpose**: Prevent data leakage when creating aggregate/contextual features  
**Last Updated**: 2025-11-13

---

## 🎯 The Golden Rule

> **If you can't calculate the feature in production WITHOUT knowing the target price, then it's LEAKAGE.**

When creating a contextual feature, always ask yourself:
- "Can I compute this feature for a NEW property without knowing its price?"
- "Does this feature require information from the current instance's target variable?"

If the answer to the second question is YES → **DO NOT CREATE THIS FEATURE**

---

## ✅ **ALLOWED Features** (LEAK-FREE)

### 1. Aggregate Statistics from Training Data

Features that summarize **historical data** from the training set:

```python
# ✅ Zone-level aggregates (from training data)
- zone_price_mean          # Average price in this zone (from train)
- zone_price_median        # Median price in this zone (from train)
- zone_price_std           # Price std dev in this zone (from train)
- zone_count               # Number of properties in this zone (from train)
- zone_surface_mean        # Average surface area in this zone (from train)

# ✅ Type × Zone aggregates (from training data)
- type_zone_price_mean     # Average price for this type in this zone (from train)
- type_zone_count          # Count of this type in this zone (from train)
- type_zone_rarity         # 1 / (count + 1) - rarity score

# ✅ Temporal aggregates (from training data)
- temporal_price_mean      # Average price in this year-month (from train)
- temporal_count           # Number of transactions in this period (from train)
```

**Why these are OK**: They use statistics computed **entirely from the training set**, without requiring knowledge of the current property's price.

---

### 2. Ratios and Interactions (No Target Involved)

Features that combine input variables without using the target:

```python
# ✅ Surface ratios (no prices involved)
- surface_vs_zone_mean     # Current surface / zone avg surface
- surface_vs_type_zone_mean # Current surface / type-zone avg surface

# ✅ Log transforms
- log_superficie           # log1p(surface)

# ✅ Interactions between inputs
- superficie_x_categoria   # surface * category_code

# ✅ Temporal features
- quarter                  # Quarter of the year (1-4)
- months_from_start        # Months since training start
```

**Why these are OK**: They use **only input features**, not the target variable of the current instance.

---

### 3. Counts and Frequencies

Features based on frequency/rarity:

```python
# ✅ Rarity scores
- type_zone_rarity         # How rare is this type-zone combination?

# ✅ Counts from training
- zone_count               # How many properties in this zone in training?
- type_zone_count          # How many of this type in this zone?
```

**Why these are OK**: They measure **how common** something is in the training data, without using target values.

---

## ❌ **PROHIBITED Features** (CAUSE LEAKAGE)

### 1. Features Using Current Instance's Target

**NEVER** create features that require the current property's price:

```python
# ❌ LEAKAGE: Requires current property's price
- price_vs_zone_mean       # current_price / zone_mean_price
- price_vs_zone_median_ratio
- price_zone_zscore        # (current_price - zone_mean) / zone_std

# ❌ LEAKAGE: Uses current price
- prezzo_mq                # current_price / surface
- prezzo_mq_vs_zone        # (price/surface) / (zone_avg_price/zone_avg_surface)

# ❌ LEAKAGE: Position in distribution requires knowing current price
- price_zone_iqr_position  # Where does current price fall in zone IQR?
- price_zone_range_position # Where does current price fall in zone range?

# ❌ LEAKAGE: Comparison with temporal mean
- price_vs_temporal_mean   # current_price / temporal_mean_price
```

**Why these are LEAKAGE**:
1. **Training**: Model sees: "This property is 20% above zone average" → learns this pattern
2. **Production**: To compute this feature, you need to know the property's price... but that's what you're trying to predict!
3. **Result**: The model essentially "sees" the answer during training

---

### 2. Features from Future Data

Never use information from data that comes **after** the current instance chronologically:

```python
# ❌ LEAKAGE: Using future data
- next_month_price_mean    # Average price in NEXT month (future!)
- future_trend             # Price trend computed using future data

# ❌ LEAKAGE: Computing stats on full dataset (train + test together)
- global_price_mean        # If computed on train+test before splitting
```

---

### 3. Target Encoding Without Proper Isolation

Be careful with target encoding of categorical variables:

```python
# ❌ LEAKAGE: Direct mean of target by category on full data
df['zone_encoded'] = df.groupby('zone')['price'].transform('mean')

# ✅ CORRECT: Fit on train only, then transform test
from category_encoders import TargetEncoder
encoder = TargetEncoder()
train_encoded = encoder.fit_transform(X_train, y_train)  # Fit on train
test_encoded = encoder.transform(X_test)  # Transform test
```

---

## 🔧 Implementation Pattern

### Correct Workflow (Fit/Transform Pattern)

```python
def fit_contextual_features(train_df, target_col='AI_Prezzo_Ridistribuito'):
    """
    Compute statistics ONLY from training data.
    Returns: Dictionary of statistics to be saved and reused.
    """
    stats = {}
    
    # ✅ Compute zone-level stats from TRAIN only
    zone_stats = train_df.groupby('AI_ZonaOmi')[target_col].agg([
        ('zone_price_mean', 'mean'),
        ('zone_price_median', 'median'),
        ('zone_count', 'count'),
    ])
    stats['zone_price'] = zone_stats.to_dict('index')
    
    return stats


def transform_contextual_features(df, stats):
    """
    Apply pre-computed statistics to any dataframe (train/val/test).
    """
    df = df.copy()
    
    # ✅ Merge pre-computed stats
    zone_price_df = pd.DataFrame.from_dict(stats['zone_price'], orient='index')
    df = df.merge(zone_price_df, left_on='AI_ZonaOmi', right_index=True, how='left')
    
    # ✅ Create ratios that DON'T use target of current instance
    df['surface_vs_zone_mean'] = df['AI_Superficie'] / (df['zone_surface_mean'] + 1e-8)
    
    # ❌ DO NOT CREATE: price_vs_zone_mean (requires current price!)
    # df['price_vs_zone_mean'] = df['AI_Prezzo_Ridistribuito'] / (df['zone_price_mean'] + 1e-8)
    
    return df


# Usage
stats = fit_contextual_features(train_df)           # Fit on train
train_transformed = transform_contextual_features(train_df, stats)
test_transformed = transform_contextual_features(test_df, stats)  # Use same stats
```

---

## 🧪 How to Test for Leakage

### Test 1: Can You Compute This in Production?

```python
def is_leak_free(feature_computation):
    """
    Mental checklist:
    1. Does it require the target value of the current instance? → LEAKAGE
    2. Does it use data from the future? → LEAKAGE
    3. Does it use test set statistics? → LEAKAGE
    4. Can I compute it in production with only input features? → OK
    """
    pass
```

### Test 2: Separate Fit and Transform

```python
def test_no_leakage():
    """Test that transform doesn't need current instance's target."""
    train = pd.DataFrame({
        'zone': ['A', 'A', 'B', 'B'],
        'price': [100, 200, 150, 250],
    })
    
    test = pd.DataFrame({
        'zone': ['A', 'B'],
        # Note: NO price column for test!
    })
    
    # ✅ Should work without test prices
    stats = fit_contextual_features(train)
    test_transformed = transform_contextual_features(test, stats)
    
    # ❌ If this fails, you have leakage!
    assert 'zone_price_mean' in test_transformed.columns
```

### Test 3: Check for Target Column Usage

```python
def test_transform_no_target_column():
    """Transform should work even if target column is missing."""
    train = pd.DataFrame({
        'zone': ['A', 'A'],
        'price': [100, 200],
        'surface': [50, 100],
    })
    
    test = pd.DataFrame({
        'zone': ['A'],
        'surface': [75],
        # NO 'price' column!
    })
    
    stats = fit_contextual_features(train, target_col='price')
    
    # ✅ Should NOT fail (transform doesn't need target)
    test_transformed = transform_contextual_features(test, stats)
```

---

## 📊 Examples

### Example 1: Zone Price Statistics

```python
# ✅ CORRECT
zone_mean = train.groupby('zone')['price'].mean()  # Compute on train
test['zone_mean_from_train'] = test['zone'].map(zone_mean)  # Apply to test

# ❌ WRONG: Leakage (uses test prices)
full_data = pd.concat([train, test])
zone_mean = full_data.groupby('zone')['price'].mean()  # Includes test!
```

### Example 2: Price per Square Meter

```python
# ❌ WRONG: Leakage (uses current property's price)
df['prezzo_mq'] = df['price'] / df['surface']
# Model learns: "properties with high price_per_sqm have high prices"
# Production: Can't compute price_per_sqm without knowing price!

# ✅ CORRECT: Use zone-level average from training
zone_prezzo_mq_mean = train['price'] / train['surface']
zone_avg = zone_prezzo_mq_mean.groupby(train['zone']).mean()
test['zone_prezzo_mq_mean'] = test['zone'].map(zone_avg)
```

### Example 3: Temporal Features

```python
# ✅ CORRECT: Historical average from training data
temporal_stats = train.groupby('year_month')['price'].mean()
test['historical_price_mean'] = test['year_month'].map(temporal_stats)

# ❌ WRONG: Including test data
all_data = pd.concat([train, test])
temporal_stats = all_data.groupby('year_month')['price'].mean()  # Leakage!
```

---

## 🚨 Common Mistakes

### Mistake 1: Computing Stats on Full Dataset

```python
# ❌ WRONG
df = load_full_data()
df['zone_mean'] = df.groupby('zone')['price'].transform('mean')
train, test = split_data(df)  # TOO LATE! Test prices already leaked into zone_mean

# ✅ CORRECT
train, test = split_data(df)  # Split FIRST
zone_mean = train.groupby('zone')['price'].mean()  # Compute on train only
train['zone_mean'] = train['zone'].map(zone_mean)
test['zone_mean'] = test['zone'].map(zone_mean)
```

### Mistake 2: Using Transform Instead of Map

```python
# ❌ WRONG: transform uses current row's target
test['zone_mean'] = test.groupby('zone')['price'].transform('mean')
# This computes mean of test set! Leakage!

# ✅ CORRECT: map uses pre-computed values
zone_mean = train.groupby('zone')['price'].mean()  # From train
test['zone_mean'] = test['zone'].map(zone_mean)  # Apply to test
```

### Mistake 3: Forgetting to Handle Unseen Categories

```python
# ⚠️ PARTIAL: What if test has zone 'C' not in train?
zone_mean = train.groupby('zone')['price'].mean()  # Only has zones A, B
test['zone_mean'] = test['zone'].map(zone_mean)  # Zone C → NaN

# ✅ BETTER: Handle unseen categories
zone_mean = train.groupby('zone')['price'].mean()
global_mean = train['price'].mean()  # Fallback
test['zone_mean'] = test['zone'].map(zone_mean).fillna(global_mean)
```

---

## ✅ Checklist Before Adding a New Feature

Before adding any contextual/aggregate feature, verify:

- [ ] ✅ Feature is computed using ONLY training data statistics
- [ ] ✅ Feature does NOT use current instance's target value
- [ ] ✅ Feature does NOT use future data (temporal leakage)
- [ ] ✅ Feature can be computed in production without knowing the target
- [ ] ✅ Transform function works even if target column is missing from input
- [ ] ✅ Fit/Transform pattern is used (fit on train, transform on test)
- [ ] ✅ Test added to verify no leakage (`test_contextual_features_no_leakage.py`)
- [ ] ✅ Documentation updated with feature description

---

## 📚 References

1. **Data Leakage in Machine Learning**: [Kaggle Blog](https://www.kaggle.com/code/alexisbcook/data-leakage)
2. **Target Encoding Best Practices**: [Towards Data Science](https://towardsdatascience.com/why-you-should-try-mean-encoding-17057262cd0)
3. **Time Series Cross-Validation**: [scikit-learn docs](https://scikit-learn.org/stable/modules/cross_validation.html#time-series-split)

---

## 🆘 Need Help?

If you're unsure whether a feature causes leakage, ask yourself:

1. **The Production Test**: "Can I compute this feature for a new property listing **before** I know its sale price?"
   - YES → Probably safe
   - NO → Leakage!

2. **The Time Travel Test**: "Does this feature require information from the future or from data I shouldn't have access to yet?"
   - YES → Leakage!
   - NO → Probably safe

3. **The Test Set Test**: "If I remove the target column from the test set, can I still compute this feature?"
   - YES → Probably safe
   - NO → Leakage!

**When in doubt**: Ask a team member or create a test to verify!

---

**Last Updated**: 2025-11-13  
**Maintainer**: ML Engineering Team  
**Status**: ✅ Active Guidelines
