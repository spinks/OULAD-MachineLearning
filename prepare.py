import pandas as pd
import sklearn
import numpy as np

# Student Info
student_info = pd.read_csv('./anonymisedData/studentInfo.csv')

# Assesment Data
assesments = pd.read_csv('./anonymisedData/assessments.csv',
                         usecols=[
                             'code_module', 'code_presentation',
                             'id_assessment', 'assessment_type'
                         ])
student_assesments = pd.read_csv(
    './anonymisedData/studentAssessment.csv',
    usecols=['id_assessment', 'id_student', 'score'])

assesments = assesments.query('assessment_type != "Exam"')

merged_assessments = student_assesments.merge(assesments, on='id_assessment')

merged_assessments = merged_assessments.groupby(
    ['id_student', 'code_module', 'code_presentation',
     'assessment_type'])['score'].mean().reset_index()

merged_assessments = (merged_assessments.set_index([
    'id_student', 'code_module', 'code_presentation'
]).pivot(columns="assessment_type")['score'].reset_index().rename_axis(None,
                                                                       axis=1))

# VLE Data
vle = pd.read_csv('./anonymisedData/vle.csv',
                  usecols=['id_site', 'activity_type'])
student_vle = pd.read_csv('./anonymisedData/studentVle.csv.gz')

merged_vle = student_vle.merge(vle, on='id_site')
merged_vle = merged_vle.groupby(
    ['id_student', 'code_module', 'code_presentation', 'activity_type'])

vle_visits = merged_vle['date'].count().reset_index()
vle_visits = (vle_visits.set_index([
    'id_student', 'code_module', 'code_presentation'
]).pivot(columns="activity_type")['date'].reset_index().rename_axis(None,
                                                                    axis=1))

vle_clicks = merged_vle['sum_click'].sum().reset_index()
vle_clicks = (vle_clicks.set_index([
    'id_student', 'code_module', 'code_presentation'
]).pivot(columns="activity_type")['sum_click'].reset_index().rename_axis(
    None, axis=1))

# Master Table
master = pd.merge(student_info,
                  merged_assessments,
                  on=['id_student', 'code_module', 'code_presentation'])
master = master.merge(
    vle_visits.add_suffix('_visits'),
    left_on=['id_student', 'code_module', 'code_presentation'],
    right_on=[
        'id_student_visits', 'code_module_visits', 'code_presentation_visits'
    ])
master = master.merge(
    vle_clicks.add_suffix('_clicks'),
    left_on=['id_student', 'code_module', 'code_presentation'],
    right_on=[
        'id_student_clicks', 'code_module_clicks', 'code_presentation_clicks'
    ])
master.drop([
    'id_student_visits', 'code_module_visits', 'code_presentation_visits',
    'id_student_clicks', 'code_module_clicks', 'code_presentation_clicks'
],
            axis=1,
            inplace=True)

# Cleaning Master Table
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

print('Master table created:', master.shape)


def split_labels(df):
    values = df.drop('final_result', axis=1)
    labels = df['final_result'].copy()
    return values, labels


from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler


def pipeline(df):
    vle_types = df.filter(
        regex='_visits$', axis=1).columns.values.tolist() + df.filter(
            regex='_clicks$', axis=1).columns.values.tolist()
    other_numeric = [
        'imd_band', 'age_band', 'num_of_prev_attempts', 'studied_credits',
        'CMA', 'TMA'
    ]

    # Imputers
    df[vle_types] = df[vle_types].fillna(0)
    df[other_numeric] = df[other_numeric].fillna(
        df[other_numeric].mean(axis=0))

    # Column Transformer
    ct = ColumnTransformer([('cat', OneHotEncoder(), [
        'code_module', 'code_presentation', 'gender', 'region',
        'highest_education'
    ]), ('std_scaler', StandardScaler(), vle_types + other_numeric)],
                           remainder='drop')

    return ct.fit_transform(df)


def prepare_labels(labels):
    lab_dict = {
        'Pass': 'Pass',
        'Fail': 'Fail',
        'Withdrawn': 'Fail',
        'Distinction': 'Pass'
    }
    return labels.replace(lab_dict)


from sklearn.externals.joblib import load, dump


def load_model(file):
    return load(file)


def save_model(file, model):
    dump(model, file)


def train_tree_classifier(values, labels):
    from sklearn.tree import DecisionTreeClassifier
    tree_class = DecisionTreeClassifier()
    tree_class.fit(values, labels)
    return tree_class


def train_forest_classifier(values, labels):
    from sklearn.ensemble import RandomForestClassifier
    forest_class = RandomForestClassifier(n_estimators=100)
    forest_class.fit(values, labels)
    return forest_class


# housing_predictions = tree_reg.predict(housing_prepared)

# Running the test
from sklearn.model_selection import train_test_split
train_set, test_set = train_test_split(master, test_size=0.2, random_state=20)

train_values, train_labels = split_labels(train_set)
test_values, test_labels = split_labels(test_set)

train_values = pipeline(train_values)
train_labels = prepare_labels(train_labels)

# forest_classifier = train_forest_classifier(train_values, train_labels)

test_values = pipeline(test_values)
test_labels = prepare_labels(test_labels)

forest_classifier = load_model('my_model.pkl')

y_predictions = (forest_classifier.predict(test_values) == 'Pass')
y_true = (test_labels == 'Pass')

from sklearn.metrics import roc_auc_score
from sklearn.metrics import precision_score, recall_score
from sklearn.metrics import accuracy_score
print('precision', precision_score(y_true, y_predictions))
print('recall', recall_score(y_true, y_predictions))
print('accuracy', accuracy_score(y_true, y_predictions))

y_predictions = forest_classifier.predict_proba(test_values)[:, 1]
print('roc', roc_auc_score(y_true, y_predictions))
