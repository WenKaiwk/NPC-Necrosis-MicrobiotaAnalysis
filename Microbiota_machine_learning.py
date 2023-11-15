"""
@Time : 2023/11/15 18:56
@Author : wenkai
@File : Microbiota_machine_learning
"""

import os
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import auc
import numpy as np


#config
inputdir = os.getcwd()

outputdir = os.path.join(inputdir, "Result") # Output path

#----------------------------------------------Data---------------------------------------------
# Import data
df_all = pd.read_excel(os.path.join(inputdir, "Data", 'base_heir.xlsx'), 0)
train_id = df_all.iloc[:, 0]
y = df_all.iloc[:, 0:3]  # label+ID
Y = df_all.iloc[:, 2]  # label
X = df_all.iloc[:, 3:5626] # Microbiota + clinical data


# -----------------------------------Data standardization---------------------------------------
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)


#--------------------------------------Feature selection----------------------------------------
def SelectLasso(k, X, Y):
    from sklearn.linear_model import LassoCV
    # your feature data is stored in the X variable and the target variable is stored in the y variable
    # Feature selection was performed using Lasso regression
    alphas = np.logspace(-3, 1, 100)
    lasso = LassoCV(cv=5, alphas=alphas)  # Select the regularization parameters using cross validation
    lasso.fit(X, Y)
    # Get feature selection results
    selected_features = np.argsort(np.abs(lasso.coef_))[::-1][:k]
    selected_columns = X.columns[selected_features]
    X_after = X[selected_columns]
    # Output the selected feature index
    return selected_columns, X_after

sl_fea, sl_after = SelectLasso(10, X_scaled, Y)


#------------------------------------Update filtered data---------------------------------------
X_index = sl_fea
X_after = X_scaled[X_index]


#------------------------------------------modeling---------------------------------------------
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.metrics import roc_auc_score
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.metrics import plot_roc_curve
from scipy import interp

# Create a list of models, including multiple classifiers
models = [
    ('Logistic Regression', LogisticRegression()),
    ('Random Forest', RandomForestClassifier(n_estimators=20, random_state=10)),
    ('SVM', SVC(kernel='rbf', probability=True, random_state=10)),
    ('AdaBoost', AdaBoostClassifier(n_estimators=50, random_state=10)),
    ('Decision Tree', DecisionTreeClassifier(random_state=10)),
    ('XGBoost', XGBClassifier(n_estimators=100, random_state=10))
]

def built_model(X_after, y, save_path, seed):
    """
    X_after: Standardized filtered feature data
    y: ID+label
    save_path: result saving path
    """
    auc_scores = {}  # Save the AUC score for each model
    # train the model, and evaluate performance
    for model_name, model in models:
        # The data set was divided by 5-fold cross-validation
        tprs = []
        aucs = []
        mean_fpr = np.linspace(0, 1, 100)
        kf = KFold(n_splits=5, random_state=seed, shuffle=True)
        fig, ax = plt.subplots()
        for i, (train_index, val_index) in enumerate(kf.split(X_after)):
            print(i + 1, "fold", train_index, val_index)
            X_train = X_after.iloc[train_index]
            X_val = X_after.iloc[val_index]
            y_train = y.iloc[train_index]
            y_val = y.iloc[val_index]
            model.fit(X_train, y_train['label'])  # Training model
            y_pred_prob = model.predict_proba(X_val)[:, 1]  # Acquisition prediction probability
            # The prediction results are tabulated
            result_list = []
            for index, (y_pred, label, Pid) in enumerate(zip(y_pred_prob, y_val["label"], y_val["id"])):
                print("id:", Pid)
                print("pred:", y_pred)
                print("label:", label)
                result_list.append([Pid, label, y_pred])
            df = pd.DataFrame(result_list, columns=["id", "label", "pred"])
            df.to_csv(os.path.join(save_path, "Result", "{}_{}fold_{}seed.csv".format(model_name, i+1, seed)))
            # Save model weight
            model_save_path = "{}_{}fold_{}seed.pkl".format(model_name, i + 1, seed)
            joblib.dump(model, os.path.join(save_path, "weight", model_save_path))

            auc_score = roc_auc_score(y_val['label'], y_pred_prob)  # Calculate the AUC score
            auc_scores[model_name, seed, i+1] = auc_score  # Save the AUC score


            # drawing pictures
            viz = plot_roc_curve(model, X_val, y_val['label'],
                                 name='ROC fold {}'.format(i),
                                 alpha=0.3, lw=1, ax=ax)
            interp_tpr = interp(mean_fpr, viz.fpr, viz.tpr)
            interp_tpr[0] = 0.0
            tprs.append(interp_tpr)
            aucs.append(viz.roc_auc)
        ax.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r',
                label='Chance', alpha=.8)

        mean_tpr = np.mean(tprs, axis=0)
        mean_tpr[-1] = 1.0
        mean_auc = auc(mean_fpr, mean_tpr)
        std_auc = np.std(aucs)
        ax.plot(mean_fpr, mean_tpr, color='b',
                label=r'Mean ROC (AUC = %0.2f $\pm$ %0.2f)' % (mean_auc, std_auc),
                lw=2, alpha=.8)

        std_tpr = np.std(tprs, axis=0)
        tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
        tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
        ax.fill_between(mean_fpr, tprs_lower, tprs_upper, color='grey', alpha=.2,
                        label=r'$\pm$ 1 std. dev.')

        ax.set(xlim=[-0.05, 1.05], ylim=[-0.05, 1.05],
               title="Receiver operating characteristic example")
        ax.legend(loc="lower right")
        plt.savefig(os.path.join("AUC_Figure", '{}_seed{}.png'.format(model_name, seed)))
        plt.close()

    # Returns the AUC score dictionary
    return auc_scores

# example
save_path = inputdir  # The path to the folder where the model weights are saved
seed = 42
auc_scores_X = built_model(X_after, y, save_path, seed)




