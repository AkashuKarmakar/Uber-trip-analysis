   Uber Trip Analysis & Prediction using Machine Learning

This project analyzes Uber ride data along with weather conditions and builds machine learning models to:

-  Predict the **ride price**
-  Classify the **cab type** (Uber or Lyft)
-  Detect **surge pricing**

---

##  Dataset Overview
   dataset link:- 'https://www.kaggle.com/datasets/brllrb/uber-and-lyft-dataset-boston-ma'
   
The dataset contains over **690,000** records with trip details and corresponding weather data. It includes:

- Trip information (hour, day, month, source, destination, cab type, distance, etc.)
- Weather information (temperature, wind speed, humidity, visibility, etc.)
- Target columns for predictions: `price`, `cab_type`, `surge_multiplier`

---

##  Preprocessing Steps

- Handled missing values in `price` column.
- Extracted time-based features from timestamp (`hour`, `day`, `month`).
- One-hot encoded categorical variables or used LabelEncoder where needed.
- Normalized numerical features if required.
- Saved individual encoders using `joblib` for later decoding.

---

##  Machine Learning Models
models link:- 'https://huggingface.co/AkashuKarmakar/Akash_UberTripAnalysis_projectModels/tree/main' 
### 1. Price Prediction (Regression)

- **Target**: `price`
- **Model Used**: `RandomForestRegressor`
- **Input Features**: All numerical + encoded categorical features
- **Output**: Estimated ride fare

---

### 2. Cab Type Classification (Multiclass Classification)

- **Target**: `cab_type`
- **Model Used**: `RandomForestClassifier`
- **Labels**: e.g., `'Uber'`, `'Lyft'`
- **Saved**: `cabtype_model.pkl`

---

### 3. Surge Pricing Detection (Binary Classification)

- **Target**: `is_surge` (1 if `surge_multiplier` > 1.0 else 0)
- **Model Used**: `LogisticRegression`
- **Output**: 0 (No surge), 1 (Surge pricing applied)

---

##  Model Saving and Loading

All models and encoders are saved using `joblib`:

```python
joblib.dump(model, 'price_model.pkl')
joblib.dump(label_encoder, 'encoders/cab_type_encoder.pkl')
