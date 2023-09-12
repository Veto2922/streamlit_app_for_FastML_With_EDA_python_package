
# These comments are for the code reader from Electropi

# Note:
# 1-FastML_With_EDA is not pandas profiling it's my own package, I made it for this project and published it in Pypi, You Can  check this and how to install it by those links:
# 1)https://pypi.org/project/FastML-With-EDA/
# 2)https://github.com/Veto2922/Fast-Machine-Learning-With-EDA-python-package
# 2) Please follow the requirements.txt file, readme in GitHub, and the messages in-app to avoid any errors

# This is a list of updates in the new code:

# 1- Automate preprocessing, detect column types , null values, and more

# 2. EDA and Preprocessing:
#    - The new code includes a checkbox for handling missing values using different strategies such as "mean," "median," "mode," or "remove."
#    - The percentage of missing values before and after handling is displayed.
#    - The option to drop selected columns from the dataset is included.

# 3. Detect and Remove Outliers:
#    - It calculates and displays the percentage of outliers in each numerical column.
#    - The option to perform a second layer of removing outliers is available.

# 4. Machine Learning:
#    - The new code offers two modes for machine learning: "By FastML" and "By pycaret."
#    - In the "By FastML" mode, we choose regression or classification tasks automatic based on the number of unique values in the target variable.
#    - You can choose from various models and customize hyperparameters.
#    -Compare model performance metrics and select the best model.

#    - In the "By pycaret" mode, the code uses the `pycaret` library for automated machine learning  and displays classification or regression metrics and plots.

# 5. User Interface:
#    - The new code features an enhanced user interface with better organization and descriptions for each section and option.

# good luck:)







import streamlit as st
import pandas as pd
from FastML_With_EDA import EDA_and_preprocessing as EP
from FastML_With_EDA import AutoML
import json
import matplotlib.pyplot as plt
# from pycaret.regression import *
# from pycaret.classification import *


# Main Streamlit app
def main():
    st.title(":blue[Welcome to FastML With EDA Web App] :computer:")

    # Upload data
    st.sidebar.header("Upload your data (CSV, XLSX, XLS, SQL)")
    uploaded_file = st.sidebar.file_uploader(
        "", type=["csv", "xlsx", "xls", "sql"])

    if uploaded_file:
        try:
            # Load data from the uploaded file
            data = pd.read_csv(uploaded_file) if uploaded_file.name.endswith('.csv') else \
                pd.read_excel(uploaded_file) if uploaded_file.name.endswith(('.xlsx', '.xls')) else \
                pd.read_sql(uploaded_file)
            st.sidebar.success("Data loaded successfully")

            # EDA and Preprocessing Options
            st.sidebar.header(
                ":blue[Exploratory Data Analysis (EDA) and Preprocessing]")


##################################################################################################################

            st.sidebar.subheader(":red[1.Explore Data:]")
            if st.sidebar.checkbox("Explore Data"):
                st.subheader("Explore Data:")
                st.write('The first 5 rows in the Data')
                st.write(data.head())
                st.write("Number of rows:", data.shape[0])
                st.write("Number of columns:", data.shape[1])
                st.write("Data Types:")
                st.write(data.dtypes)

                # # Display unique values in categorical columns
                # categorical_columns = data.select_dtypes(include=['object'])
                # if not categorical_columns.empty:
                #     st.header("Unique Values in Categorical Columns:")
                #     for column in categorical_columns.columns:
                #         unique_values = data[column].nunique()
                #         st.write(f"{column}: {unique_values} unique values")

                # Display missing values after handling
                st.write("Missing Values percentage before Handling:")
                st.write(data.isnull().sum() / 100)
                st.markdown(
                    """<hr style="height:10px;border:20;color:#ffff;background-color:#ffff;" /> """, unsafe_allow_html=True)

 ##################################################################################################################

            st.sidebar.subheader(":red[2.Handle Missing Values]")
            if st.sidebar.checkbox("Handle Missing Values"):
                st.subheader("Handle Missing Values:")
                missing_strategy = st.selectbox("Select Missing Value Handling Strategy", [
                                                'None', "mean", "median", 'mode',  "remove"])
                if missing_strategy == 'mode':
                    missing_strategy = 'mean'

                data = EP.handling_missing_values(data, missing_strategy)
                st.success(
                    'Note: We are dropping the rows with missing values <3% and dropping the columns with missing values >90 automatic')
                st.success('If you select the missing Value Handling Strategy by mean or median the numerical data are filled by mean or median and categorical data are filled by Mode(most frequent) Automatically')

                # Display missing values after handling
                st.write("Missing Values percentage after Handling:")
                st.write(data.isnull().sum() / 100)

                if missing_strategy == 'remove':
                    st.write('Data shape after remove missing values')
                    st.write("Number of rows:", data.shape[0])
                    st.write("Number of columns:", data.shape[1])
                st.markdown(
                    """<hr style="height:10px;border:20;color:#ffff;background-color:#ffff;" /> """, unsafe_allow_html=True)

##################################################################################################################

             # Detect and Remove Outliers
            st.sidebar.header(":red[3.Detect and Remove Outliers:]")
            detect_outliers = st.sidebar.checkbox("Detect Outliers")
            outliers_percentage = {}

            def out_pre(data):
                for col in data.select_dtypes(include=["int64", "float64"]):
                    q1 = data[col].quantile(0.25)
                    q3 = data[col].quantile(0.75)
                    iqr = q3 - q1
                    lower_bound = q1 - 1.5 * iqr
                    upper_bound = q3 + 1.5 * iqr
                    outlier = data[(data[col] < lower_bound) |
                                   (data[col] > upper_bound)]
                    outlier_percentage = (len(outlier) / len(data)) * 100
                    outliers_percentage[col] = outlier_percentage
                    # st.write(f"{col}: {outlier_percentage:.2f}%")
                st.dataframe(outliers_percentage)

            if detect_outliers:
                st.subheader("Percentage of Outliers in Each Column:")

                out_pre(data)
                st.write('Data shape before remove outliers')
                st.write("Number of rows:", data.shape[0])
                st.write("Number of columns:", data.shape[1])

            if st.sidebar.checkbox("Remove Outliers"):
                data = EP.outliers(data, precentage=False, remove=True)
                st.success("Outliers removed successfully!")
                # outliers
                # Display the percentage of outliers again after removal
                st.subheader(
                    "Percentage of Outliers in Each Column After Removal:")
                out_pre(data)
                st.write('Data shape after remove outliers')
                st.write("Number of rows:", data.shape[0])
                st.write("Number of columns:", data.shape[1])
                st.subheader(
                    "Do you want to make 2nd layer of removing outliers?")
                if st.checkbox("Yes"):
                    data = EP.outliers(data, precentage=False, remove=True)
                    st.success("2nd layer Outliers removed successfully!")
                    st.subheader(
                        "Percentage of Outliers in Each Column After 2nd Removal:")
                    out_pre(data)
                    st.write('Data shape after 2nd remove outliers')
                    st.write("Number of rows:", data.shape[0])
                    st.write("Number of columns:", data.shape[1])
                st.markdown(
                    """<hr style="height:10px;border:20;color:#ffff;background-color:#ffff;" /> """, unsafe_allow_html=True)


##################################################################################################################

            st.sidebar.subheader(":red[4.Do you want to drop any columns?]")
            if st.sidebar.checkbox("Yes."):
                st.subheader("Drop columns:")
                columns = data.columns
                col_to_drop = st.multiselect(
                    "Select columns you want to drop: ", columns)
                if col_to_drop:
                    data = data.drop(col_to_drop, axis=1)
                    st.success(f"drop {col_to_drop} columns  done")
                    st.write(data.head())
                    st.write("Number of rows:", data.shape[0])
                    st.write("Number of columns:", data.shape[1])
            st.markdown(
                """<hr style="height:10px;border:20;color:#ffff;background-color:#ffff;" /> """, unsafe_allow_html=True)


##################################################################################################################

            st.sidebar.subheader(":red[5.Data Visualization]")
            if st.sidebar.checkbox("Data Visualization"):
                st.subheader("Data Visualization:")
                plot_types = st.selectbox("Select Plot Types", [
                                          "histogram", "kde", "ecdf", "regression", "pairplot", "scatter", "line", "box", "count", "bar", "point"])
                fig = plt.figure(figsize=(5, 6))
                if plot_types:
                    x = st.selectbox("Select X-axis", data.columns)
                    if "histogram" in plot_types or "kde" in plot_types:
                        EP.plot_data(data, plot_types, x)
                        st.pyplot(fig)
                    else:
                        y = st.selectbox("Select Y-axis", data.columns)
                        EP.plot_data(data, plot_types, x, y)
                        st.pyplot(fig)
                st.markdown(
                    """<hr style="height:10px;border:20;color:#ffff;background-color:#ffff;" /> """, unsafe_allow_html=True)

##################################################################################################################

            st.sidebar.subheader(":red[6.Encode Categorical Features]")
            if st.sidebar.checkbox("label"):
                st.subheader("Encode Categorical Features:")
                data = AutoML.encode_categorical_features(
                    data, type_of_encoding='label')
                st.success(f'Label Encodeing id done')
                st.write(data.head())
            elif st.sidebar.checkbox("onehot"):
                data = AutoML.encode_categorical_features(
                    data, type_of_encoding='label')
                st.success(f'Onehot Encodeing id done')
                st.write(data.head())
            st.markdown(
                """<hr style="height:10px;border:20;color:#ffff;background-color:#ffff;" /> """, unsafe_allow_html=True)


##################################################################################################################

            # Machine Learning Options
            # st.header("Machine Learning" )

            st.sidebar.header(':blue[Supervised Machine Learning] :rocket:')
##################################################################################################################

            if st.sidebar.checkbox("By FastML"):
                st.header(
                    ':blue[Supervised Machine Learning with FastML] :rocket:')
                st.warning(
                    "Please make sure you remove outliers, handle missing values, and encode all categorical columns. If you don't, you may encounter errors.")

                st.subheader("Select Target Variable")
                target_variable = st.selectbox(
                    "Select Target Variable", data.columns)

                task = list(data[target_variable].value_counts())

                if len(task) > 20:
                    task_type = "regression"
                    st.success('Your task type is: Regression')
                    model_name = st.selectbox("Select Regression Model to Train (note: all = try all models)", [
                                              'all', "linear", "svm", "knn", "decision_tree", "random_forest", "boosting", "xgboost"])

                elif len(task) >= 2 and len(task) <= 20:
                    task_type = "classification"
                    st.success('Your task type is: Classification')
                    model_name = st.selectbox("Select Classification Model to Train (note: all = try all models)", [
                                              'all', "logistic", "svm", "knn", "decision_tree", "random_forest", "boosting", "xgboost"])

                # st.subheader("Train Machine Learning Models")
                # task_type = st.radio("Select Task Type", ["regression", "classification"])
                test_z = st.slider('Select the Percentage of test size', 0, 70)
                scaling = st.selectbox("Select Feature Scaling", [
                                       "None", "minmax", "standard"])
                # model_name = st.selectbox("Select Models to Train (note: all = try all models)", ['all', "linear", "logistic", "svm", "knn", "decision_tree", "random_forest", "boosting", "xgboost"])

                if model_name == 'all':
                    models, training_df, testing_df, best_model = AutoML.train_machine_learning_models(
                        data, target_column=target_variable, task_type=task_type, test_size=test_z/100, scaling=scaling
                    )

                    if task_type == "regression":
                        st.subheader("Model Comparison")
                        st.write("Training Metrics")
                        st.write(training_df)
                        st.write("Testing Metrics")
                        st.write(testing_df)
                        st.subheader("Best Model")
                        st.write(best_model)
                    else:
                        del training_df["Confusion Matrix"]
                        del testing_df["Confusion Matrix"]
                        st.subheader("Model Comparison")
                        st.write("Training Metrics")
                        st.write(training_df)
                        st.write("Testing Metrics")
                        st.write(testing_df)
                        st.subheader("Best Model")

                        st.write(best_model)

                else:
                    st.write(
                        'Do you want to specify hyperparameters for your model?')
                    if st.checkbox('No'):
                        models, training_df, testing_df = AutoML.train_machine_learning_models(
                            data, target_column=target_variable, task_type=task_type, model_names=model_name, test_size=test_z/100, scaling=scaling
                        )
                        # del training_df["Confusion Matrix"]
                        # del testing_df["Confusion Matrix"]
                        # st.subheader(f"{model_name} Model")
                        st.write("Training Metrics")
                        st.write(training_df)
                        # st.dataframe(training_df)
                        st.write("Testing Metrics")
                        st.write(training_df)
                        # st.dataframe(testing_df)
                    if st.checkbox('Yes. '):
                        st.warning(
                            'This feature is under development, and may still not work')
                        hyper_parameters_str = st.text_input(
                            'Write your model hyperparameters in a dict form like {"l1": 0.1, "k": 5}')
                        try:
                            hyper = json.loads(hyper_parameters_str)

                            models, training_df, testing_df = AutoML.train_machine_learning_models(
                                data, target_column=target_variable, task_type=task_type, model_names=model_name,
                                test_size=test_z/100, hyperparameters=hyper, scaling=scaling
                            )

                            st.subheader(f"{model_name} Model")
                            st.write("Training Metrics")
                            st.write(training_df)
                            st.write("Testing Metrics")
                            st.write(testing_df)
                        except json.JSONDecodeError as e:
                            st.error(f"Error parsing hyperparameters: {e}")

##################################################################################################################

            elif st.sidebar.checkbox("By pycaret"):
                st.header(
                    ':blue[Supervised Machine Learning with pycaret] :rocket:')
                # st.warning("Please make sure you remove outliers, handle missing values, and encode all categorical columns. If you don't, you may encounter errors.")

                st.subheader("Select Target Variable")
                target_variable = st.selectbox(
                    "Select Target Variable", data.columns)

                task = list(data[target_variable].value_counts())

                if len(task) > 20:
                    st.success(
                        'Your task type is: Regression (may take some time)')

                    from pycaret import regression

                    # exp = regression.setup(
                    #     data, target=target_variable, session_id=123)

                    # # Find the best regression model
                    # best_model = regression.compare_models(n_select=3)

                    # # Display the best regression model details
                    # st.subheader("Best 3 Regression Model")
                    # st.write(best_model)

                    # Set silent to true to disable asking questions.
                    s = regression.setup(data, target=target_variable)

                    best = regression.compare_models()  # get best model
                    regression.save_model(best, 'my_best_pipeline')

                    st.write('#### Metrics')
                    st.dataframe(regression.pull(), height=200)

                    # Analyzes the performance of a trained model on the test set.
                    # Only available in Notebook.
                    # evaluate_model(best)
                    st.subheader("Best Regression Model")
                    st.write(best)
                    # Plots
                    regression.plot_model(
                        best, plot='residuals', display_format='streamlit')
                    regression.plot_model(
                        best, plot='feature_all', display_format='streamlit')
                    regression.plot_model(
                        best, plot='error', display_format='streamlit')

                    # Predicts label on the holdout set.
                    pred_holdout = regression.predict_model(best)
                    st.write('#### Predictions from holdout set')
                    st.dataframe(pred_holdout, height=200)

                    # Predicts label on the data.
                    # predictions = predict_model(best, data=data)

                    # Test the saved model.
                    loaded_model = regression.load_model('my_best_pipeline')
                    predictions = regression.predict_model(
                        loaded_model, data=data)
                    st.write('#### Predictions from data set')
                    st.dataframe(predictions, height=200)

                elif len(task) >= 2 and len(task) <= 20:
                    st.success(
                        'Your task type is: Classification (may take some time)')
                    from pycaret import classification

                    best = classification.compare_models()  # get best model
                    classification.save_model(best, 'my_best_pipeline')

                    st.write('#### Metrics')
                    st.dataframe(classification.pull(), height=200)

                    # Analyzes the performance of a trained model on the test set.
                    # Only available in Notebook.
                    # evaluate_model(best)
                    st.subheader("Best classification Model")
                    st.write(best)
                    # Plots

                    classification.plot_model(
                        best, plot='confusion_matrix', display_format='streamlit')
                    classification.plot_model(
                        best, plot='class_report', display_format='streamlit')
                    classification.plot_model(
                        best, plot='feature_all', display_format='streamlit')
                    classification.plot_model(
                        best, plot='error', display_format='streamlit')
                    classification.plot_model(
                        best, plot='auc', display_format='streamlit')

                    # Predicts label on the holdout set.
                    pred_holdout = classification.predict_model(best)
                    st.write('#### Predictions from holdout set')
                    st.dataframe(pred_holdout, height=200)

                    # Predicts label on the data.
                    # predictions = classification.predict_model(best, data=data)

                    # Test the saved model.
                    # data.columns = data.columns.str.replace(' ', '')
                    loaded_model = classification.load_model(
                        'my_best_pipeline')
                    predictions = classification.predict_model(
                        loaded_model, data=data)
                    st.write('#### Predictions from data set')
                    st.dataframe(data)

        except Exception as e:
            st.error(f"Error: {str(e)}")


if __name__ == "__main__":
    main()
