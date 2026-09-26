# House Price Predictor

## 📌 Overview
A machine learning model that predicts house prices based on property features such as square footage, number of bedrooms/bathrooms, and number of offers received, using Linear Regression.

## 🎯 Problem
Estimating a fair market price for a house is a common real-world need for buyers, sellers, and real estate agents. This project uses historical housing data to learn the relationship between property features and sale price.

## 📂 Dataset
- **Features used:**
  - `SqFt`: square footage of the house
  - `Bedrooms`: number of bedrooms
  - `Bathrooms`: number of bathrooms
  - `Offers`: number of offers received on the house
  - (also available in the raw data but not used as features: `Brick`, `Neighborhood`)
- **Target:** `Price`

## 🛠️ Tech Stack & Approach
| Step | Tool | Description |
|---|---|---|
| Data loading | `pandas` | Reading the housing CSV file |
| Train/test split | `train_test_split` | 85% training / 15% testing |
| Model | `LinearRegression` | Predicts a continuous price value from property features |
| Evaluation | `model.score()` | R² score measuring how well the model explains price variation |

## ⚙️ How to Run
```bash
pip install pandas numpy scikit-learn
```

```python
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

# Load data
df = pd.read_csv("home.csv")

# Select features and target
x = df[['SqFt', 'Bedrooms', 'Bathrooms', 'Offers']]
y = df['Price']

# Train/test split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.15, train_size=0.85, random_state=2)

# Train model
model = LinearRegression()
model.fit(x_train, y_train)

# Evaluate
print(model.score(x_test, y_test))
```

## 📊 Results
R² score on the test set (after removing the leaked feature, see note below): to be measured.

## ⚠️ Important Note / Limitations
- **Data leakage:** in the current notebook, the `Price` column (the target) was accidentally included in the feature set `x` as well. This causes the model to "see" the answer during training, resulting in an artificially perfect score (1.0). Make sure `Price` is excluded from `x` before training, as shown in the corrected code above.
- The dataset is small, so the model may not generalize well to houses very different from those in the training data.
- Categorical features like `Brick` and `Neighborhood` were not used in this version and could improve accuracy if properly encoded and included.

## ✍️ Author
Atila Eslami
