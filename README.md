# 🍽️ Restaurant Rating Prediction Model

This project focuses on predicting restaurant ratings using machine learning techniques.  
It uses data preprocessing, feature engineering, and a regression model to estimate ratings based on various restaurant attributes.

---

## 📌 Project Overview

The goal of this project is to:
- Clean and preprocess real-world restaurant data
- Handle missing and inconsistent values
- Convert categorical data into numerical format
- Train a machine learning model to predict ratings
- Evaluate model performance using standard metrics

---

## 🛠️ Technologies Used

- Python
- Pandas
- Scikit-learn

---

## 📂 Dataset

- Dataset used: `zomato.csv`
- Contains restaurant-related information such as:
  - Location
  - Cuisine
  - Reviews
  - Ratings

---

## ⚙️ Data Preprocessing

The following steps were performed:

- Removed unnecessary columns:
  - `url`, `address`, `phone`, `reviews_list`, `menu_item`, `dish_liked`
- Handled missing values
- Cleaned rating column (removed `/5` and converted to numeric)
- Dropped rows with invalid or missing ratings
- Converted categorical columns using **Label Encoding**

---

## 🤖 Model Used

- **Linear Regression**

---

## 🔍 Workflow

1. Load dataset
2. Clean and preprocess data
3. Encode categorical features
4. Split dataset (80% training, 20% testing)
5. Train model
6. Predict ratings
7. Evaluate performance

---

## 📊 Evaluation Metrics

- Mean Squared Error (MSE)
- R² Score

---

## 📈 Output

The model prints:
- Mean Squared Error
- R² Score

---

## 🚀 Future Improvements

- Use advanced models (Random Forest, XGBoost)
- Improve feature engineering
- Use better encoding techniques
- Hyperparameter tuning

---

## 📌 Conclusion

This project demonstrates a complete machine learning pipeline:
- Data cleaning
- Feature engineering
- Model training
- Evaluation

It serves as a strong foundation for building real-world prediction systems.
