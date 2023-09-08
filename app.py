import streamlit as st
import pandas as pd
from FastML_With_EDA.EDA_and_preprocessing import *
from FastML_With_EDA.AutoML import *
import json







# Main Streamlit app
def main():
    st.title("FastML With EDA Web App")

    # Upload data
    st.sidebar.header("Upload your data (CSV, XLSX, XLS, SQL)")
    uploaded_file = st.sidebar.file_uploader("", type=["csv", "xlsx", "xls", "sql"])
    
    if uploaded_file:
        try:
            # Load data from the uploaded file
            data = pd.read_csv(uploaded_file) if uploaded_file.name.endswith('.csv') else \
                   pd.read_excel(uploaded_file) if uploaded_file.name.endswith(('.xlsx', '.xls')) else \
                   pd.read_sql(uploaded_file)
            st.sidebar.success("Data loaded successfully!")

            

            # EDA and Preprocessing Options
            st.sidebar.header(":blue[Exploratory Data Analysis (EDA) and Preprocessing]")

            st.sidebar.subheader("1. Explore Data:")
            if st.sidebar.checkbox("Explore Data"):
                st.write('The first 5 rows in the Data')
                st.write(data.head())
                st.write("Number of rows:", data.shape[0])
                st.write("Number of columns:", data.shape[1])
                st.write("Data Types:")
                st.write(data.dtypes)
                
                # Display unique values in categorical columns
                categorical_columns = data.select_dtypes(include=['object'])
                if not categorical_columns.empty:
                    st.header("Unique Values in Categorical Columns:")
                    for column in categorical_columns.columns:
                        unique_values = data[column].nunique()
                        st.write(f"{column}: {unique_values} unique values")

                # Display missing values after handling
                st.write("Missing Values percentage before Handling:")
                st.write(data.isnull().sum() / 100 )




            
            st.sidebar.subheader("2. Handle Missing Values")
            if st.sidebar.checkbox("Handle Missing Values"):
                missing_strategy = st.selectbox("Select Missing Value Handling Strategy", ['None',"mean", "median", "remove"])
                data = handling_missing_values(data, missing_strategy)
                st.success('Note: We are dropping the rows with missing values <3% and dropping the columns with missing values >90 automatic')

                # Display missing values after handling
                st.write("Missing Values percentage after Handling:")
                st.write(data.isnull().sum() / 100 )


            
             # Detect and Remove Outliers
            st.sidebar.header("3. Detect and Remove Outliers:")
            detect_outliers = st.sidebar.checkbox("Detect Outliers")
            outliers_percentage = {}
            def out_pre(data):
                    for col in data.select_dtypes(include=["int64", "float64"]):
                        q1 = data[col].quantile(0.25)
                        q3 = data[col].quantile(0.75)
                        iqr = q3 - q1
                        lower_bound = q1 - 1.5 * iqr
                        upper_bound = q3 + 1.5 * iqr
                        outlier = data[(data[col] < lower_bound) | (data[col] > upper_bound)]
                        outlier_percentage = (len(outlier) / len(data)) * 100
                        outliers_percentage[col] = outlier_percentage
                        st.write(f"{col}: {outlier_percentage:.2f}%")

            if detect_outliers:
                st.subheader("Percentage of Outliers in Each Column:")
                
                out_pre(data)

            if st.sidebar.checkbox("Remove Outliers"):
                data = outliers(data, precentage=False, remove=True)
                st.success("Outliers removed successfully!")
                # outliers
                # Display the percentage of outliers again after removal
                st.subheader("Percentage of Outliers in Each Column After Removal:")
                out_pre(data)

            # st.subheader("3. Detect and Remove Outliers")
            # if st.checkbox("Detect and Remove Outliers"):
                
            #     st.write(data)




            
            
            st.sidebar.subheader("4.Data Visualization")
            if st.sidebar.checkbox("Data Visualization"):
                plot_types = st.selectbox("Select Plot Types", ["histogram", "kde", "ecdf", "regression", "pairplot", "scatter", "line", "box", "count", "bar", "point"])
                fig = plt.figure(figsize=(5,6))
                if plot_types:
                    x = st.selectbox("Select X-axis", data.columns)
                    if "histogram" in plot_types or "kde" in plot_types:
                        plot_data(data, plot_types, x)
                        st.pyplot(fig)
                    else:
                        y = st.selectbox("Select Y-axis", data.columns)
                        plot_data(data, plot_types, x, y)
                        st.pyplot(fig)


            
            st.sidebar.subheader("5. Encode Categorical Features")
            if st.sidebar.checkbox("label"):
                    data = encode_categorical_features(data, type_of_encoding = 'label')
                    st.write(f'Label Encodeing id done')
                    st.write(data.head())
            elif st.sidebar.checkbox("onehot"):
                    data = encode_categorical_features(data, type_of_encoding = 'label')
                    st.write(f'Onehot Encodeing id done')
                    st.write(data.head())




            # Machine Learning Options
            # st.header("Machine Learning" )
            
            st.sidebar.header(':blue[Machine Learning] :rocket:')
            if st.sidebar.checkbox("Start"):
                st.warning("Please make sure you remove outliers, handle missing values, and encode all categorical columns. If you don't, you may encounter errors.")

                st.subheader("Select Target Variable")
                target_variable = st.selectbox("Select Target Variable", data.columns)
                st.subheader("Train Machine Learning Models")
                task_type = st.radio("Select Task Type", ["regression", "classification"])
                test_z = st.slider('Select the Percentage of test size', 0, 70)
                scaling = st.selectbox("Select Feature Scaling", ["None", "minmax", "standard"])
                model_name = st.selectbox("Select Models to Train (note: all = try all models)", ['all', "linear", "logistic", "svm", "knn", "decision_tree", "random_forest", "boosting", "xgboost"])

                if model_name == 'all':
                    models, training_df, testing_df, best_model = train_machine_learning_models(
                        data, target_column=target_variable, task_type=task_type, test_size=test_z/100, scaling=scaling
                    )
                    st.subheader("Model Comparison")
                    st.write("Training Metrics")
                    st.write(training_df)
                    st.write("Testing Metrics")
                    st.write(testing_df)
                    st.subheader("Best Model")
                    st.write(best_model)
                else:
                    st.write('Do you want to specify hyperparameters for your model?')
                    if st.checkbox('No'):
                        models, training_df, testing_df = train_machine_learning_models(
                            data, target_column=target_variable, task_type=task_type, model_names=model_name, test_size=test_z/100, scaling=scaling
                        )
                        st.subheader(f"{model_name} Model")
                        st.write("Training Metrics")
                        st.write(training_df)
                        st.write("Testing Metrics")
                        st.write(testing_df )
                    if st.checkbox('Yes'):
                        st.warning('This feature is under development, and may still not work')
                        hyper_parameters_str = st.text_input('Write your model hyperparameters in a dict form like {"l1": 0.1, "k": 5}')
                        try:
                            hyper = json.loads(hyper_parameters_str)
                            
                            models, training_df, testing_df = train_machine_learning_models(
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
                    
                                
                    
        except Exception as e:
            st.error(f"Error: {str(e)}")

if __name__ == "__main__":
    main()









# if st.checkbox('Yes'):
#                         st.warning('This feature is under development, and may still not work')
                        
#                         # Load the predefined hyperparameters for the selected model
#                         predefined_hyperparameters = get_predefined_hyperparameters(model_name)
                        
#                         # Create a dictionary to store user-defined hyperparameters
#                         user_hyperparameters = {}
                        
#                         # Display hyperparameter inputs for the user
#                         st.subheader(f"Specify Hyperparameters for {model_name}")
#                         for param_name, param_value in predefined_hyperparameters.items():
#                             user_value = st.number_input(param_name, min_value=0.0, max_value=1.0, value=param_value, step=0.01)
#                             user_hyperparameters[param_name] = user_value
                        
#                         if st.button('Train Model'):
#                             try:
#                                 models, training_df, testing_df = train_machine_learning_models(
#                                     data, target_column=target_variable, task_type=task_type, model_names=model_name,
#                                     test_size=test_z/100, hyperparameters={model_name: user_hyperparameters}, scaling=scaling
#                                 )

#                                 st.subheader(f"{model_name} Model")
#                                 st.write("Training Metrics")
#                                 st.write(training_df)
#                                 st.write("Testing Metrics")
#                                 st.write(testing_df)
#                             except Exception as e:
#                                 st.error(f"Error training model: {e}")


# def get_predefined_hyperparameters(model_name):
#     # Define predefined hyperparameters for each model
#     predefined_hyperparameters = {
#         "linear": {
#             "alpha": [0.1, 0.01, 0.001],
#             "l1_ratio": [0.1, 0.5, 0.9],
#             # Add other hyperparameters for the linear model here
#         },
#         "logistic": {
#             "C": [1.0, 0.1, 0.01],
#             "penalty": ["l1", "l2"],
#             # Add other hyperparameters for the logistic model here
#         },
#         "svm": {
#             "C": [1.0, 0.1, 0.01],
#             "kernel": ["linear", "rbf", "poly"],
#             # Add other hyperparameters for the SVM model here
#         },
#         "knn": {
#             "n_neighbors": [3, 5, 7],
#             "weights": ["uniform", "distance"],
#             # Add other hyperparameters for the KNN model here
#         },
#         "decision_tree": {
#             "criterion": ["gini", "entropy"],
#             "max_depth": [None, 10, 20],
#             # Add other hyperparameters for the decision tree model here
#         },
#         "random_forest": {
#             "n_estimators": [100, 200, 300],
#             "max_depth": [None, 10, 20],
#             # Add other hyperparameters for the random forest model here
#         },
#         "boosting": {
#             "n_estimators": [50, 100, 200],
#             "learning_rate": [0.01, 0.1, 0.2],
#             # Add other hyperparameters for the boosting model here
#         },
#         "xgboost": {
#             "n_estimators": [50, 100, 200],
#             "learning_rate": [0.01, 0.1, 0.2],
#             # Add other hyperparameters for the xgboost model here
#         },
#         # Define hyperparameters for other models as needed
#     }

#     # Return predefined hyperparameters for the specified model
#     return predefined_hyperparameters.get(model_name, {})
