# + tags=["parameters"]
# declare a list tasks whose products you want to use as inputs
upstream = None

# +
import os
from pathlib import Path
import joblib
from sklearn.metrics import DetCurveDisplay, RocCurveDisplay, f1_score
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import xgboost as xgb
import seaborn as sns
import joblib
import matplotlib.pyplot as plt
import imgkit
from sklearn.metrics import (
    classification_report,
    accuracy_score,
    f1_score,
    make_scorer,
    confusion_matrix
)
from sklearn.model_selection import (
    train_test_split,
    cross_val_score,
    StratifiedKFold,
    RandomizedSearchCV,
)
from pathlib import Path
from imblearn.pipeline import Pipeline
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import SMOTE
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.utils import estimator_html_repr

def generate_pipeline(X):
    """
    Generates the pipeline with preprocessing and a classifier.

    Args:
        X (pandas.DataFrame): The input data.

    Returns:
        A RandomizedSearchCV object.
    """

    # Define the undersampler and oversampler
    undersampler = RandomUnderSampler(random_state=12)
    oversampler = SMOTE(random_state=12)

    # Define the classifier
    clf = xgb.XGBClassifier(
        n_jobs=-1,
        random_state=42,
    )

    # Define the pipeline
    pipeline = Pipeline(
        steps=[
            ("preprocessor", None),
            ("undersampler", undersampler),
            ("oversampler", oversampler),
            ("classifier", clf),
        ]
    )

    # Define the parameter grid for hyperparameter tuning
    parameters = {
        "preprocessor": [ColumnTransformer(transformers=[("num", StandardScaler(), 
                                                          X.select_dtypes(exclude=["object"]).columns.tolist()), 
                                                         ("cat", OneHotEncoder(handle_unknown='ignore'), 
                                                          X.select_dtypes(include=["object"]).columns.tolist())])],
        "undersampler__sampling_strategy": ['majority', 'not minority', 'not majority', 'all'],
        "classifier__n_estimators": [50, 75, 100],
        "classifier__learning_rate": [0.01, 0.05, 0.1],
        "classifier__max_depth": [3, 5, 7],
        "classifier__subsample": [0.5, 0.8, 1],
        "classifier__colsample_bytree": [0.5, 0.8, 1],
        "classifier__reg_alpha": [0, 0.1, 1],
        "classifier__reg_lambda": [0, 0.1, 1],
        "classifier__scale_pos_weight": [1, 5, 10],
    }

    # Create the RandomizedSearchCV object
    clf = RandomizedSearchCV(
        pipeline,
        parameters,
        n_iter=100,  # Number of parameter settings that are sampled. n_iter trades off runtime vs quality of the solution.
        cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42),
        scoring="f1_weighted",
        n_jobs=-1,
        verbose=1,
        random_state=42
    )

    return clf


def perform_cross_validation(X, y, pipeline, cv=5, scoring="f1_weighted", success_metric=0.81):
    f1_scorer = make_scorer(f1_score, average=scoring)
    cv_scores = cross_val_score(
        pipeline, X, y, cv=cv, scoring=f1_scorer, n_jobs=-1
    )
    avg_f1_score = round(cv_scores.mean(), 2)

    if avg_f1_score >= success_metric:
        print("Success: The average F1 score is above or equal to the success metric.")
    else:
        print("Failure: The average F1 score is below the success metric.")

    return avg_f1_score

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

    # Save the feature importances plot to a file
    generate_feature_importances(pipeline, X)

# +
def roc_curve_save_plot(pipeline, X_test, y_test):
    """
    This function saves the ROC and DET curves to a file

    Args:
        pipeline (sklearn.pipeline.Pipeline): Pipeline with preprocessing,and the classifier
        X_test (pandas.DataFrame): The test data.
        y_test (pandas.Series): The test labels.
    """

    # Plot the ROC and DET curves
    fig, [ax_roc, ax_det] = plt.subplots(1, 2, figsize=(11, 5))


    RocCurveDisplay.from_estimator(pipeline, X_test, y_test, ax=ax_roc)
    DetCurveDisplay.from_estimator(pipeline, X_test, y_test, ax=ax_det)

    ax_roc.set_title("Receiver Operating Characteristic (ROC) curves")
    ax_det.set_title("Detection Error Tradeoff (DET) curves")

    plt.legend()

    plt.savefig(os.path.abspath(os.path.join(os.getcwd(), 'reports', 'figures', "roc.png")))



# -

def generate_feature_importances(pipeline, X):
    """
    This function generates the feature importances plot

    Args:
        pipeline (sklearn.pipeline.Pipeline): Pipeline with preprocessing,and the classifier
        X (pandas.DataFrame): The input data.

    Returns:
        A GridSearchCV object.
    """

    importances = pipeline.named_steps['classifier'].feature_importances_
    numerical_feature_names = X.select_dtypes(exclude=['object']).columns.tolist()
    categorical_feature_names = pipeline.named_steps['preprocessor'].named_transformers_['cat'].get_feature_names_out(input_features=X.select_dtypes(include=['object']).columns.tolist())
    feature_names = numerical_feature_names + categorical_feature_names.tolist()

    feature_importances = pd.DataFrame({'feature': feature_names, 'importance': importances})
    feature_importances = feature_importances.sort_values('importance', ascending=False).reset_index(drop=True)

    plt.figure(figsize=(10, 10))
    sns.barplot(data=feature_importances, x='importance', y='feature')
    plt.title('Feature Importances')
    plt.savefig(os.path.abspath(os.path.join(os.getcwd(), 'reports', 'figures', 'feature-importances.png')))

if __name__=="__main__":

    csv_path = os.path.abspath(os.path.join(os.getcwd(),  'data', 'raw', 'term-deposit-marketing-2020.csv'))
    data = pd.read_csv(csv_path)

     # drop the rows that have a z-score greater than 3 for only class '0'
    #data = data[(z < 3).all(axis=1) | (data['y'] != 0)]
    data['y'] = data['y'].apply(lambda x: 1 if x == 'yes' else 0)

    # Define feature columns and target column
    X = data.drop(columns=['y'])
    y = data['y']

    
    # Split the data into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, stratify=y, random_state=12)

    tune_model = generate_pipeline(X)
    # Train the pipeline
    tune_model.fit(X_train, y_train)

    best_model = tune_model.best_estimator_
    best_score = tune_model.best_score_
    y_pred = best_model.predict(X_test)
    
    # Printing results
    print("Best parameters:", tune_model.best_params_)
    print("Cross-validated accuracy score on training data: {:0.4f}".format(tune_model.best_score_))
    print()
    
    print("Weighted average F1 score", f1_score(y_test, y_pred, average='weighted'))
    print("Macro average F1 score", f1_score(y_test, y_pred, average='macro'))
    print("Micro average F1 score", f1_score(y_test, y_pred, average='micro'))

     # Evaluate the model
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("Classification Report:\n", classification_report(y_test, y_pred))

    with open(os.path.abspath(os.path.join(os.getcwd(), 'reports', "results.txt")), "w") as file:
        file.write("Best parameters: {}\n".format(tune_model.best_params_))
        file.write("Cross-validated accuracy score on training data: {:0.4f}\n\n".format(tune_model.best_score_))
        
        file.write("Weighted average F1 score {}\n".format(f1_score(y_test, y_pred, average='weighted')))
        file.write("Macro average F1 score {}\n".format(f1_score(y_test, y_pred, average='macro')))
        file.write("Micro average F1 score {}\n\n".format(f1_score(y_test, y_pred, average='micro')))

        file.write("Accuracy: {}\n".format(accuracy_score(y_test, y_pred)))
        file.write("Classification Report:\n{}\n".format(classification_report(y_test, y_pred)))

    file.close()

    # Generate confusion matrix
    plt.figure(figsize=(10, 10))
    fig = confusion_matrix(y_test, y_pred)
    plt.savefig(os.path.abspath(os.path.join(os.getcwd(), 'reports', 'figures', "confusion-matrix.png")))

    # # Generate ROC curve
    roc_curve_save_plot(best_model, X_test, y_test)
    
    # # Save the model
    save_model(best_model)


