# + tags=["parameters"]
# declare a list tasks whose products you want to use as inputs
upstream = None


# +
import pandas as pd
from sklearn.svm import SVC
import os
import xgboost as xgb
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.metrics import classification_report, accuracy_score
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_score
from pathlib import Path
from sklearn.utils import estimator_html_repr
import imgkit
import joblib
import imgkit
from sklearn.metrics import DetCurveDisplay, RocCurveDisplay, f1_score
import matplotlib.pyplot as plt
import seaborn as sns
from imblearn.under_sampling import RandomUnderSampler
from sklearn.ensemble import VotingClassifier, RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from imblearn.over_sampling import SMOTE
from sklearn.base import BaseEstimator, TransformerMixin


def generate_pipeline(X, model):
    """
    
    This function generates a pipeline with preprocessing, oversampling, and the classifier.

    Args:
        X (pandas.DataFrame): DataFrame containing the features
        model (sklearn model): Model to be used for classification

    Returns:
        pipeline (sklearn.pipeline.Pipeline): Pipeline with preprocessing, oversampling, and the classifier
    """
    
    # Identify categorical and numerical columns
    categorical_cols = X.select_dtypes(include=['object']).columns.tolist()
    numerical_cols = X.select_dtypes(exclude=['object']).columns.tolist()

    # Define transformers for categorical and numerical columns
    categorical_transformer = OneHotEncoder(handle_unknown='ignore')
    numerical_transformer = StandardScaler()


    preprocessor = ColumnTransformer(
            transformers=[
                ('num', numerical_transformer, numerical_cols),
                ('cat', categorical_transformer, categorical_cols),
            ])


    # create the pipeline with the preprocessor, oversampling, and the classifier
    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', model)
    ])
        
    return pipeline



# +
def perform_cross_validation(X, y, pipeline, cv=5, scoring='f1_weighted', success_metric=0.81):

    """

    This function performs cross-validation and prints the average accuracy.

    Args:
        X (pandas.DataFrame): DataFrame containing the features
        y (pandas.Series): Series containing the target variable
        pipeline (sklearn.pipeline.Pipeline): Pipeline with preprocessing,and the classifier
        cv (int): Number of folds for cross-validation
        scoring (str): Metric to be used for cross-validation

    """

    # Perform 5-fold cross-validation
    cv_scores = cross_val_score(pipeline, X, y, cv=cv, scoring=scoring)

    # Calculate the average accuracy
    avg_accuracy = round(cv_scores.mean(),2)

    print("5-fold Cross-validation F1-score:", avg_accuracy)

    if avg_accuracy >= success_metric:
        print("Success: The average accuracy is above or equal to the success metric.")
    else:
        print("Failure: The average accuracy is below the success metric.")

# +
def save_model(pipeline):

    """

    This function saves the pipeline diagram and the pipeline sklearn object into the models folder

    Args:
        pipeline (sklearn.pipeline.Pipeline): Pipeline with preprocessing,and the classifier

    """

    # Save the pipeline diagram to a file
    result_path = os.path.abspath(os.path.join(os.getcwd(), 'models'))
    with open(Path(result_path, "pipeline_diagram.html"), "w") as f:
        f.write(estimator_html_repr(pipeline))

    # Convert the HTML to an image
    imgkit.from_file(str(Path(result_path, "pipeline_diagram.html")), str(Path(result_path, "pipeline_diagram.png")))

    # Save the pipeline to a file
    joblib.dump(pipeline, Path(result_path,"pipeline.joblib"))

# +
def roc_curve_save_plot(pipeline, X_test, y_test):
    """
    
    This function generates a ROC curve and saves it to the figures folder.

    Args:
        pipeline (sklearn.pipeline.Pipeline): Pipeline with preprocessing,and the classifier
        X_test (pandas.DataFrame): DataFrame containing the features of the test set
        y_test (pandas.Series): Series containing the target variable of the test set

    """

    fig, [ax_roc, ax_det] = plt.subplots(1, 2, figsize=(11, 5))

    RocCurveDisplay.from_estimator(pipeline, X_test, y_test, ax=ax_roc)
    DetCurveDisplay.from_estimator(pipeline, X_test, y_test, ax=ax_det)

    ax_roc.set_title("Receiver Operating Characteristic (ROC) curves")
    ax_det.set_title("Detection Error Tradeoff (DET) curves")

    plt.legend()

    # save the plot as a file
    plt.savefig(os.path.abspath(os.path.join(os.getcwd(), 'reports', 'figures', "roc")))


# -

def generate_feature_importances(pipeline,X, X_train, y_train):
    """
    
    This function generates a bar plot of feature importances and saves it to the figures folder.

    Args:
        pipeline (sklearn.pipeline.Pipeline): Pipeline with preprocessing,and the classifier
        X (pandas.DataFrame): DataFrame containing the features
        X_train (pandas.DataFrame): DataFrame containing the features of the training set
        y_train (pandas.Series): Series containing the target variable of the training set

    """


    # Get feature importances
    importances = pipeline.named_steps['classifier'].feature_importances_

    # Get feature names
    numerical_feature_names = X.select_dtypes(exclude=['object']).columns.tolist()
    categorical_feature_names = pipeline.named_steps['preprocessor'].named_transformers_['cat'].get_feature_names_out(input_features=X.select_dtypes(include=['object']).columns.tolist())

    feature_names = numerical_feature_names + categorical_feature_names.tolist()

    # Create a dataframe with feature names and importances
    feature_importances = pd.DataFrame({'feature': feature_names, 'importance': importances})

    # Sort the dataframe by feature importance
    feature_importances = feature_importances.sort_values('importance', ascending=False).reset_index(drop=True)


    # Plot the feature importances
    plt.figure(figsize=(10, 10))
    sns.barplot(data=feature_importances, x='importance', y='feature')
    plt.title('Feature Importances')
    plt.savefig(os.path.abspath(os.path.join(os.getcwd(), 'reports', 'figures', 'feature-importances.png')))

if __name__=="__main__":

    csv_path = os.path.abspath(os.path.join(os.getcwd(),  'data', 'raw', 'term-deposit-marketing-2020.csv'))
    data = pd.read_csv(csv_path)

    # Prepare target variable
    data['y'] = data['y'].apply(lambda x: 1 if x == 'yes' else 0)

    # Define feature columns and target column
    X = data.drop(columns=['y'])
    y = data['y']

    # Split the data into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    # Define the undersampler and oversampler
    undersampler = RandomUnderSampler(random_state=42)
    # Create the SMOTE transformer
    oversampler = SMOTE(random_state=42)
    
    # Undersample the majority class
    X_resampled, y_resampled = undersampler.fit_resample(X_train, y_train)

 
    # Create a pipeline with preprocessing, oversampling, and the classifier
    # Set up model pipeline
    # Define the cost matrix
    cost_matrix = [[0, 1], [10, 0]]
    clf1 = KNeighborsClassifier(2)
    clf2 = SVC(random_state=42, probability=True, kernel='linear', class_weight='balanced')
    clf3 = RandomForestClassifier(random_state=42)
    clf4 = xgb.XGBClassifier(random_state=42)
    classifiers = {
                   "KNN": clf1, 
                   "SVM": clf2,
                   "RFC": clf3,
                   "XGB": clf4
                }

    eclf1 = VotingClassifier(estimators=[
                                        ('knn', clf1), 
                                         ('svm', clf2), 
                                         ('dt', clf3),
                                         ('xgb',clf4)], voting='soft')


    pipeline = generate_pipeline(X_resampled, eclf1)
    # Train the pipeline
    pipeline.fit(X_resampled, y_resampled)
    

    y_pred = pipeline.predict(X_test)
    print("Weighted average F1 score", f1_score(y_test, y_pred, average='weighted'))
    print("Macro average F1 score", f1_score(y_test, y_pred, average='macro'))
    print("Micro average F1 score", f1_score(y_test, y_pred, average='micro'))


    # Perform cross-validation
    perform_cross_validation(X_resampled, y_resampled, pipeline, cv=5, scoring='f1', success_metric=0.81)
    
    # Fit the model
    pipeline.fit(X_resampled, y_resampled)

    # Make predictions on the test set
    y_pred = pipeline.predict(X_test)

    # Evaluate the model
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("Classification Report:\n", classification_report(y_test, y_pred))

    # # Generate ROC curve
    roc_curve_save_plot(pipeline, X_test, y_test)
    
    # # Save the model
    save_model(pipeline)


