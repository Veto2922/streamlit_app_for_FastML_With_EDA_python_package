# streamlit_app_for_FastML_With_EDA_python_package


# FastML With EDA Web App

FastML With EDA Web App is a user-friendly web application built with Streamlit that complements the FastML_With_EDA Python package. It empowers users to perform exploratory data analysis (EDA), data preprocessing, and machine learning model training effortlessly. Whether you are a data scientist, analyst, or machine learning enthusiast, this web app simplifies your data-driven tasks.

![FastMl web app](https://github.com/Veto2922/streamlit_app_for_FastML_With_EDA_python_package/assets/114834171/411c3849-af85-4dbe-82bb-dd6062224f0e)


## Features

- **Data Loading:** Easily upload and load datasets in various formats, including CSV, Excel, XLSX, XLS, and SQL databases.
- **Exploratory Data Analysis (EDA):** Gain insights into your dataset with summary statistics, data type analysis, unique values in categorical columns, and missing value identification.
- **Missing Value Handling:** Choose from multiple strategies for handling missing data, such as mean or median imputation or complete removal of rows/columns.
- **Outlier Detection:** Detect and optionally remove outliers from numerical columns.
- **Data Visualization:** Generate a wide range of data visualizations, including histograms, scatter plots, regression plots, and more, to better understand your data.
- **Categorical Encoding:** Encode categorical features using Label Encoding or One-Hot Encoding.
- **AutoML Model Training:** Train machine learning models for both regression and classification tasks with optional feature scaling. It supports various algorithms, including linear regression, logistic regression, SVM, k-nearest neighbors, decision trees, random forests, gradient boosting, and XGBoost.
- **Model Evaluation:** Evaluate trained models with common metrics like R2-score, MSE, RMSE, MAE for regression, and Precision, Recall, F1-score, Confusion Matrix for classification.
- **Model Selection:** Automatically select the best-performing model based on your chosen evaluation metric.

## Usage

1. **Upload Data:** Begin by uploading your dataset (CSV, Excel, XLSX, XLS, or SQL).

2. **Exploration:** Explore your data, identify unique values in categorical columns, and check for missing values.

3. **Missing Value Handling:** Choose a strategy to handle missing values.

4. **Outlier Detection:** Detect and remove outliers if needed.

5. **Data Visualization:** Create various data visualizations to explore relationships in your data.

6. **Categorical Encoding:** Encode categorical features for machine learning.

7. **Machine Learning:** Train machine learning models for regression or classification tasks. You can choose from various models and customize hyperparameters.

8. **Model Comparison:** Compare model performance metrics and select the best model.


## Installation

This web app is built on top of the FastML_With_EDA Python package. To use it, you can install the package with pip:

```bash
pip install FastML-With-EDA
```

## Getting Started

To run the FastML With EDA Web App, follow these steps:

1. Clone this GitHub repository to your local machine.

2. Install the required Python packages using `pip install -r requirements.txt`.

3. Run the Streamlit app by executing `streamlit run app.py`.

4. Upload your dataset and start exploring and building machine learning models.

## List of Updates in AppV2

1. **Automated Preprocessing**
   - The new code automates preprocessing tasks, including the detection of column types, null values, and more.

2. **EDA and Preprocessing:**
   - The code now includes a checkbox for handling missing values using different strategies such as "mean," "median," "mode," or "remove."
   - It displays the percentage of missing values before and after handling.
   - An option to drop selected columns from the dataset is now available.

3. **Detect and Remove Outliers:**
   - It calculates and displays the percentage of outliers in each numerical column.
   - An option to perform a second layer of removing outliers has been added.

4. **Machine Learning:**
   - The new code offers two modes for machine learning: "By FastML" and "By pycaret."
   - In the "By FastML" mode, it automatically chooses between regression or classification tasks based on the number of unique values in the target variable.
   - Users can select from various models and customize hyperparameters.
   - Model performance metrics are compared, and the best model can be selected.

   - In the "By pycaret" mode, the code utilizes the `pycaret` library for automated machine learning and displays classification or regression metrics and plots.

5. **User Interface:**
   - The new code features an enhanced user interface with improved organization and descriptions for each section and option.

## Contributing

Contributions to this project are welcome! If you have any suggestions, bug reports, or feature requests, please open an issue or submit a pull request.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.



[FastML_With_EDA package GitHub Repository](https://github.com/Veto2922/Fast-Machine-Learning-With-EDA-python-package)

[FastML_With_EDA PyPI Package](https://pypi.org/project/FastML-With-EDA/)



