# House Price Prediction using Machine Learning

This repository contains a machine learning project that predicts house prices using the fetch_california_housing Price Dataset. The project implements an XGBoost Regressor model and covers key steps of a typical machine learning workflow.

## Project Goal

The primary goal of this project is to build a predictive system that can estimate house prices based on various features of a house, treating it as a regression problem.

## Technologies Used

*   **Python:** The core programming language.
*   **Libraries:**
    *   **NumPy:** For numerical operations, especially with arrays.
    *   **Pandas:** For data manipulation and creating data frames.
    *   **Matplotlib:** For data visualization (plotting graphs and charts).
    *   **Seaborn:** For enhanced data visualization, particularly heatmaps.
    *   **Scikit-learn (sklearn):** A comprehensive machine learning library used for:
        *   Loading the dataset (fetch_california_housing).
        *   Splitting data into training and testing sets (`train_test_split`).
        *   Evaluation metrics (`r2_score`, `mean_absolute_error`).
    *   **XGBoost:** For implementing the XGBoost Regressor model.

## Workflow

The project follows these steps:

1.  **Import Dependencies:** Import necessary Python libraries (NumPy, Pandas, Matplotlib, Seaborn, scikit-learn, XGBoost).
2.  **Data Collection & Loading:** Load the fetch_claifornia_housing Dataset using `sklearn.datasets`.
3.  **Data Pre-processing:**
    *   Load the dataset into a Pandas DataFrame.
    *   Add the 'Price' (target) column to the DataFrame.
    *   Check for the number of rows and columns.
    *   Identify and handle missing values (no missing values found in this dataset).
    *   Generate statistical measures of the dataset (`.describe()`).
4.  **Exploratory Data Analysis (EDA):**
    *   Understand the correlation between various features using a **heatmap**. This helps visualize positive and negative correlations.
5.  **Data Splitting:**
    *   Separate the features (X) from the target variable (y - 'Price').
    *   Split the dataset into training and testing data using `train_test_split` (80% training, 20% testing).
6.  **Model Training:**
    *   Initialize the **XGBoost Regressor** model.
    *   Train the model using the training data (`X_train` and `Y_train`).
7.  **Model Evaluation:**
    *   Make predictions on both the training and testing datasets.
    *   Evaluate the model's performance using regression metrics:
        *   **R-squared Error** (`r2_score`)
        *   **Mean Absolute Error (MAE)** (`mean_absolute_error`)
    *   Visualize the actual prices vs. predicted prices using a scatter plot to observe the model's accuracy.

## How to Run

1.  **Clone the repository:**
    `git clone `
2.  **Navigate to the project directory:**
    `cd house-price-prediction`
3.  **Install dependencies:**
    `pip install numpy pandas matplotlib seaborn scikit-learn xgboost`
4.  **Run the Jupyter Notebook (or Python script):**
    The code is designed to be run in a Google Colaboratory environment, but can also be executed as a standard Python script or in a Jupyter Notebook.
    `jupyter notebook project.ipynb` (if saved as a notebook)
    `python project.py` (if saved as a Python script)

## Dataset

The project uses the **Boston House Price Dataset**, which is available through scikit-learn's `datasets` module. This dataset contains 506 entries and 14 features (including the price). For more details on the dataset features, you can refer to the UCI Machine Learning Repository or Kaggle.
