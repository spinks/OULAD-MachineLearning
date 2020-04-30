import pandas as pd
import numpy as np

# tune controls wether to run the grid search on hyperparameters
# it is off by default as it takes quite a while
tune = False
# load controls wether the saved models are read in from a file
# these files are created by the tuning process
# so once you have tuned you no longer need to re-run the grid search
load = True

# Student Info
student_info = pd.read_csv('./anonymisedData/studentInfo.csv')
student_info = student_info.query('final_result != "Withdrawn"')

# Assesment Data
# Read in only useful columns to be merged
assesments = pd.read_csv('./anonymisedData/assessments.csv',
                         usecols=[
                             'code_module', 'code_presentation',
                             'id_assessment', 'assessment_type'
                         ])
student_assesments = pd.read_csv(
    './anonymisedData/studentAssessment.csv',
    usecols=['id_assessment', 'id_student', 'score'])

merged_assessments = student_assesments.merge(assesments, on='id_assessment')

# Group assesments by id_student, code_module and code_presentation such that they
# have a one to one correlation with the student_info table
merged_assessments = merged_assessments.groupby(
    ['id_student', 'code_module', 'code_presentation',
     'assessment_type'])['score'].mean().reset_index()

# Create pivot table to merge assesment type and score columns into two
# (one for both CMA and TMA) on a unique student/course row
merged_assessments = (merged_assessments.set_index([
    'id_student', 'code_module', 'code_presentation'
]).pivot(columns="assessment_type")['score'].reset_index().rename_axis(None,
                                                                       axis=1))

# VLE Data
# Read in only relevant columns from generic vle table
vle = pd.read_csv('./anonymisedData/vle.csv',
                  usecols=['id_site', 'activity_type'])
# Student data
student_vle = pd.read_csv('./anonymisedData/studentVle.csv.gz')

# Merge the activity types onto student VLE
merged_vle = student_vle.merge(vle, on='id_site')
# Again grouby student/module/presentation to be one to one with student info
merged_vle = merged_vle.groupby(
    ['id_student', 'code_module', 'code_presentation', 'activity_type'])

# Unique visits to each activity type by counting unique days
vle_uniq_visits = merged_vle['date'].count().reset_index()
# Pivot table here to turn visits column into individual columns for
# each activity with values as the number of visits
vle_uniq_visits = (vle_uniq_visits.set_index([
    'id_student', 'code_module', 'code_presentation'
]).pivot(columns="activity_type")['date'].reset_index().rename_axis(None,
                                                                    axis=1))

# Total number of interactions with each activity type
vle_interactions = merged_vle['sum_click'].sum().reset_index()
# Again pivot table to turn interactions columns into individual columns for
# each activity
vle_interactions = (vle_interactions.set_index([
    'id_student', 'code_module', 'code_presentation'
]).pivot(columns="activity_type")['sum_click'].reset_index().rename_axis(
    None, axis=1))

# Master Table
# Merge student info and prepared assesment information first
master = pd.merge(student_info,
                  merged_assessments,
                  on=['id_student', 'code_module', 'code_presentation'])
# Merge on vle_visit table adding suffix to differentiate from interactiosns table
master = master.merge(
    vle_uniq_visits.add_suffix('_uniq_visits'),
    left_on=['id_student', 'code_module', 'code_presentation'],
    right_on=[
        'id_student_uniq_visits', 'code_module_uniq_visits',
        'code_presentation_uniq_visits'
    ])
# Merge on vle_interactions
master = master.merge(
    vle_interactions.add_suffix('_interactions'),
    left_on=['id_student', 'code_module', 'code_presentation'],
    right_on=[
        'id_student_interactions', 'code_module_interactions',
        'code_presentation_interactions'
    ])
# Drop redundant rows from suffixed merges of vle table
master.drop([
    'id_student_uniq_visits', 'code_module_uniq_visits',
    'code_presentation_uniq_visits', 'id_student_interactions',
    'code_module_interactions', 'code_presentation_interactions'
],
            axis=1,
            inplace=True)

# Cleaning Master Table
# These operations replace the numeric categorical columns with a value in
# the middle of the range represented for use in the models
imd_dict = {
    '0-10%': 5,
    '10-20': 15,
    '10-20%': 15,
    '20-30%': 25,
    '30-40%': 35,
    '40-50%': 45,
    '50-60%': 55,
    '60-70%': 65,
    '70-80%': 75,
    '80-90%': 90,
    '90-100%': 95
}
age_dict = {'0-35': 17.5, '35-55': 45, '55<=': 82.5}

master.replace({"age_band": age_dict, "imd_band": imd_dict}, inplace=True)
master.query('final_result != "Withdrawn"', inplace=True)

print('Master table created:', master.shape, '\n')


def split_labels(df):
    # Remove the final result column from a table and return both
    values = df.drop('final_result', axis=1)
    labels = df['final_result'].copy()
    return values, labels


# Pipelines for standardising the table for models
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler


def pipeline(df):
    # VLE data is selected separately from other numeric as the fill strategy
    # needs to be different
    vle_types = df.filter(
        regex='_uniq_visits$', axis=1).columns.values.tolist() + df.filter(
            regex='_interactions$', axis=1).columns.values.tolist()
    # All other numeric columns
    other_numeric = [
        'imd_band', 'age_band', 'num_of_prev_attempts', 'studied_credits',
        'CMA', 'TMA'
    ]

    # Imputers
    # VLE data is filled with 0s as this is what the data truely represents
    # If a value is NA then the user hasnt interacted with/visited that activity
    df[vle_types] = df[vle_types].fillna(0)
    # Other numeric data is filled with mean
    df[other_numeric] = df[other_numeric].fillna(
        df[other_numeric].mean(axis=0))

    # Column Transformer
    # One hot encodes the remaining categorical data
    # Standard scales all numeric data
    ct = ColumnTransformer([('cat', OneHotEncoder(), [
        'code_module', 'code_presentation', 'gender', 'region',
        'highest_education'
    ]), ('std_scaler', StandardScaler(), vle_types + other_numeric)],
                           remainder='drop')

    return ct.fit_transform(df)


def prepare_labels(labels):
    # As we are only precicting pass fail we relabel disctinction as
    # pass and withdraw as fail
    # We use 1 to represent pass and 0 for fail for the scoring metric functions
    lab_dict = {'Pass': 1, 'Fail': 0, 'Withdrawn': 0, 'Distinction': 1}
    return labels.replace(lab_dict)


from joblib import load, dump


def load_model(file):
    return load(file)


def save_model(file, model):
    dump(model, file)


def train_tree_classifier(values, labels):
    # simple fit of a decision tree clasifier
    from sklearn.tree import DecisionTreeClassifier
    tree_class = DecisionTreeClassifier()
    tree_class.fit(values, labels)
    return tree_class


def train_forest_classifier(values, labels):
    # simple fit of a random forest classifier
    from sklearn.ensemble import RandomForestClassifier
    forest_class = RandomForestClassifier(n_estimators=100)
    forest_class.fit(values, labels)
    return forest_class


# Create a train test split
from sklearn.model_selection import train_test_split
train_set, test_set = train_test_split(master, test_size=0.2, random_state=20)

# Split off labels
train_values, train_labels = split_labels(train_set)
test_values, test_labels = split_labels(test_set)

# Pipeline the data to prepare for training and testing
train_values = pipeline(train_values)
train_labels = prepare_labels(train_labels)
test_values = pipeline(test_values)
test_labels = prepare_labels(test_labels)

# Train classifiers
tree_classifier = train_tree_classifier(train_values, train_labels)
forest_classifier = train_forest_classifier(train_values, train_labels)

from sklearn.model_selection import cross_val_score
print('\nTree Classifier Cross Validation Scores')
for test in ['accuracy', 'recall', 'f1', 'roc_auc']:
    scores = cross_val_score(tree_classifier,
                             train_values,
                             train_labels,
                             scoring=test,
                             cv=5)
    print(test, np.mean(scores))
print('\nRandom Forest Classifier Cross Validation Scores')
for test in ['accuracy', 'recall', 'f1', 'roc_auc']:
    scores = cross_val_score(forest_classifier,
                             train_values,
                             train_labels,
                             scoring=test,
                             cv=5)
    print(test, np.mean(scores))
if tune:
    # Random grid search for hyperparameter tuning
    from sklearn.model_selection import RandomizedSearchCV
    random_grid_decision = {
        'max_features': ['sqrt', 'log2'],
        'max_depth': [None, 20, 40, 60, 80, 100, 120],
        'min_samples_split': [2, 4, 8],
        'min_samples_leaf': [1, 2, 4],
    }
    # First we perform the decision tree optimisation
    search = RandomizedSearchCV(tree_classifier,
                                param_distributions=random_grid_decision,
                                n_iter=75,
                                cv=5,
                                n_jobs=-1,
                                scoring='roc_auc',
                                random_state=20)
    search.fit(train_values, train_labels)
    # Save the best model
    tree_classifier = search.best_estimator_
    save_model('best_tree', tree_classifier)

    # And the random forest optimisation
    random_grid_forest = {
        'n_estimators': [100, 200, 400, 600, 800, 1000, 1200],
        'max_features': ['sqrt', 'log2'],
        'max_depth': [None, 20, 40, 60, 80, 100, 120],
        'min_samples_split': [2, 4, 8],
        'min_samples_leaf': [1, 2, 4],
        'bootstrap': [True, False]
    }
    # First we perform the decision tree optimisation
    search = RandomizedSearchCV(forest_classifier,
                                param_distributions=random_grid_forest,
                                n_iter=75,
                                cv=5,
                                n_jobs=-1,
                                scoring='roc_auc',
                                random_state=20)
    search.fit(train_values, train_labels)
    # Save the best model
    forest_classifier = search.best_estimator_
    save_model('best_forest', forest_classifier)
elif load:
    tree_classifier = load_model('best_tree')
    forest_classifier = load_model('best_forest')
# Test Performance
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_auc_score
from sklearn.metrics import accuracy_score, recall_score, f1_score

# Decision tree
predictions = tree_classifier.predict(test_values)
print('\nDecision Tree Confusion Matrix\n',
      confusion_matrix(test_labels, predictions))

# Forest
predictions = forest_classifier.predict(test_values)
print('\nForest Confusion Matrix\n', confusion_matrix(test_labels,
                                                      predictions))

# Print testing scores
print('\nDecision Tree Test Scores\n')
predictions = tree_classifier.predict(test_values)
predictions_proba = tree_classifier.predict_proba(test_values)[:, 1]
print('Accuracy', accuracy_score(test_labels, predictions))
print('Recall', recall_score(test_labels, predictions))
print('F1', f1_score(test_labels, predictions))
print('ROC AUC', roc_auc_score(test_labels, predictions_proba))

print('\nRandom Forest Test Scores\n')
predictions = forest_classifier.predict(test_values)
predictions_proba = forest_classifier.predict_proba(test_values)[:, 1]
print('Accuracy', accuracy_score(test_labels, predictions))
print('Recall', recall_score(test_labels, predictions))
print('F1', f1_score(test_labels, predictions))
print('ROC AUC', roc_auc_score(test_labels, predictions_proba))

import matplotlib.pyplot as plt

from sklearn.metrics import roc_curve, auc

y_score = forest_classifier.predict_proba(test_values)[:, 1]
# Compute ROC curve and ROC area for each class
fpr = dict()
tpr = dict()
roc_auc = dict()
fpr[1], tpr[1], _ = roc_curve(test_labels, y_score)
roc_auc[1] = auc(fpr[1], tpr[1])

# Compute micro-average ROC curve and ROC area
fpr["micro"], tpr["micro"], _ = roc_curve(test_labels.ravel(), y_score.ravel())
roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])
plt.figure()
lw = 2
plt.plot(fpr[1],
         tpr[1],
         color='darkorange',
         lw=lw,
         label='Forest ROC curve (area = %0.2f)' % roc_auc[1])
y_score = tree_classifier.predict_proba(test_values)[:, 1]
# Compute ROC curve and ROC area for each class
fpr = dict()
tpr = dict()
roc_auc = dict()
fpr[1], tpr[1], _ = roc_curve(test_labels, y_score)
roc_auc[1] = auc(fpr[1], tpr[1])

# Compute micro-average ROC curve and ROC area
fpr["micro"], tpr["micro"], _ = roc_curve(test_labels.ravel(), y_score.ravel())
roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])
lw = 2
plt.plot(fpr[1],
         tpr[1],
         color='green',
         lw=lw,
         label='Decision Tree ROC curve (area = %0.2f)' % roc_auc[1])
plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curves')
plt.legend(loc="lower right")
plt.show()
