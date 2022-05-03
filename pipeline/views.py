import os
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'flight_fatal.settings')

from django.shortcuts import render, redirect
from django.http import JsonResponse, HttpResponse
from rest_framework.response import Response
from rest_framework.decorators import api_view, permission_classes, authentication_classes
from rest_framework.permissions import IsAuthenticated
from rest_framework.authentication import SessionAuthentication

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss
from sklearn.metrics import confusion_matrix
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, train_test_split, RandomizedSearchCV
from sklearn.preprocessing import MinMaxScaler
import lightgbm as lgbm
import joblib
import os
import seaborn as sns
import matplotlib.pyplot as plt
import neurokit2 as nk
from biosppy.signals import  ecg
import matplotlib as mpl
mpl.rcParams['agg.path.chunksize'] = 10000
import warnings
warnings.filterwarnings("ignore")


@api_view(['GET', 'POST'])
def pipeline(request):
    """
    This function train/test LGBM models
    parameters:
              testing(bool): True/False
              test_data(dataframe): pandas dataframe
              training(bool)L True/False
    Returns: None
    """

    # train = request.FILES["train"]
    # test = request.FILES.get("test", None)
    # testing = False if request.POST['testing'].lower() == 'false' else True
    # test_data = request.POST['test_data']
    # training = request.POST['training']
    # test_data_is_df = request.POST['test_data_is_df']
    testing = False
    test_data = None
    training = False
    test_data_is_df = False
    context = dict()
    context['result'] = {"msg": 'failed', "predicted class" : "unknown"}
    try:
        if request.method == 'POST':
            checked_box = request.POST.getlist('ch_bx')
            for i in checked_box:
                if i == "is_test_data_dataframe": test_data_is_df = True
                if i == "testing": testing = True
                if i == "training": training = True

            if testing or training:
                d = {'A': 0, 'B': 1, 'C': 2, 'D': 3}
                d_ = {0: 'A', 1: 'B', 2: 'C', 3: 'D'}
                if training:
                    print("\n<start>Loading dataset....")
                    raw = pd.read_csv("dataset/train.csv")
                    print("<complete>Loading dataset.\n")

                    Y = raw['event']
                    X = raw.drop('event', axis=1)

                    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=.2, shuffle=False)
                    print(f"X_train.shape:{X_train.shape}")
                    print(f"y_train.shape:{y_train.shape}")
                    print(f"X_test.shape:{X_test.shape}")
                    print(f"y_test.shape:{y_test.shape}")

                    y_train_num = np.array(list(map(lambda x: d[x], y_train)))
                    y_test_num = np.array(list(map(lambda x: d[x], y_test)))

                    X_train['experiment'] = X_train['experiment'].astype('category')
                    X_test['experiment'] = X_test['experiment'].astype('category')

                    train = lgbm.Dataset(X_train, label=y_train_num, categorical_feature=[1])
                    test = lgbm.Dataset(X_test, label=y_test_num, categorical_feature=[1])

                    params = {
                        "objective": "multiclass",  # used for multiclass softmax classifier
                        "metric": "multi_error",  # Error rate for multiclass classification
                        "boosting": 'gbdt',  # Using Gardient Boosted Decision Trees
                        'num_class': 4,  # Number of desired output classes is 4
                        "num_leaves": 30,  # Number of leaves in Tree based algorithms
                        "learning_rate": 0.01,
                        "bagging_fraction": 0.9,  # This is randomly select 90% of data without resampling\
                        # it will decrease impact of high variance on data
                        "bagging_seed": 0,  # Random seeds for bagging
                        "num_threads": 4,
                        "colsample_bytree": 0.5,  # Subsampling fraction for feature
                        'min_data_in_leaf': 100,  # Threshold on Data in a leaf
                        'min_split_gain': 0.00019  # Minmimum gain threshold for splitting the node
                    }

                    print('\n<start>Training Light Gradient Boosting Machine...')
                    model = lgbm.train(params,
                                       train_set=train,
                                       num_boost_round=1000,
                                       early_stopping_rounds=200,
                                       verbose_eval=200,
                                       valid_sets=[train, test]
                                       )
                    print('<complete>Training\n')

                    print("\n<start>Prediction...")
                    y_train_pred = model.predict(X_train)
                    y_test_pred = model.predict(X_test)
                    print("<complete>Prediction\n")

                    print("\n<start>LOG-LOSS calculation...")
                    print(f"Train log_loss:{log_loss(y_train, y_train_pred)}")
                    print(f"Test log_loss:{log_loss(y_test, y_test_pred)}")
                    print("<complete>LOG-LOSS calculation\n")

                    conf_mat_val1 = confusion_matrix(np.argmax(y_train_pred, axis=1), y_train_num)
                    conf_mat_val2 = confusion_matrix(np.argmax(y_test_pred, axis=1), y_test_num)

                    ax = plt.axes()
                    sns.heatmap(conf_mat_val1, annot=True, fmt="d", cmap='Blues')
                    ax.set_xlabel('predicted', fontsize=10)
                    ax.set_ylabel('actual', fontsize=10)
                    ax.set_title("Train confusion matrix")
                    ax.set_xticklabels(['A', 'B', 'C', 'D'])
                    ax.set_yticklabels(['A', 'B', 'C', 'D'], rotation="horizontal")
                    plt.show()

                    ax = plt.axes()
                    sns.heatmap(conf_mat_val2, annot=True, fmt="d", cmap='Blues')
                    ax.set_xlabel('predicted', fontsize=10)
                    ax.set_ylabel('actual', fontsize=10)
                    ax.set_title("Test confusion matrix")
                    ax.set_xticklabels(['A', 'B', 'C', 'D'])
                    ax.set_yticklabels(['A', 'B', 'C', 'D'], rotation="horizontal")
                    plt.show()

                    print("====Dumping model====")
                    joblib.dump(model, "model_final")

                if testing:
                    print("\n<start>Testing on provided dataset...")
                    model = joblib.load("C:/Users/Kira/PycharmProjects/flight_fatal/pipeline/model_final") if not training else model
                    cols = ["id", 'crew', 'experiment', 'time', 'seat', 'eeg_fp1', 'eeg_f7', 'eeg_f8',
                            'eeg_t4', 'eeg_t6', 'eeg_t5', 'eeg_t3', 'eeg_fp2', 'eeg_o1', 'eeg_p3',
                            'eeg_pz', 'eeg_f3', 'eeg_fz', 'eeg_f4', 'eeg_c4', 'eeg_p4', 'eeg_poz',
                            'eeg_c3', 'eeg_cz', 'eeg_o2', 'ecg', 'r', 'gsr']

                    dtypes = ["int"] * 2 + ["str"] + ["float"] + ["int"] + ["float"] * 23
                    test_data = request.POST['test_data']
                    print(f"TEST DATA:{test_data}")
                    if not test_data_is_df:
                        if type(test_data) == type("s"):
                            col_dtype = list(zip(cols, dtypes))
                            myarray = np.array([test_data.split(",")])
                            record = np.array(list(map(tuple, myarray)), dtype=col_dtype)
                            test_data = pd.DataFrame.from_records(record)
                        else:
                            print("<Error-3>Something went wrong...Returning!")
                            return Response(context)
                        print("Test data string being converted into test dataframe!")

                    if 'id' in test_data.columns:
                        test_df = test_data.drop(['id'], axis=1)
                    if 'event' in test_data.columns:
                        test_df = test_data.drop(['event'], axis=1)
                    test_df['experiment'] = test_df['experiment'].astype('category')
                    test_pred = model.predict(test_df)
                    pred_idx = np.argmax(test_pred, axis=-1)
                    pred_classes = list(map(lambda x: d_[x], pred_idx))
                    print(pred_classes[:10])
                    print("<complete>Testing on provided dataset...\n")
                    context['result'] = {"msg": 'success', "predicted class": pred_classes[:10][0]}
                    return Response(context)
                else:
                    print("No testing?...fine...Returning...!")

            else:
                print("<Error-2>Something went wrong...Returning!")
                return Response(context)
        return render(request, "index.html", context)

    except Exception as e:
        return Response(context)





