# usage of config:
# Input:  file_path for input
# Output_dir: file_path for output
# Max_feature: k     for top k features
# top_k_features: file_path for top_k_features output
# Choose_top_k : method for ranking features
# Dot_plot: file_path for top_k_features output
# subplot1: k         for the subplot1's x axis ranking k
# subplot2: k         for the subplot2's x axis ranking k
# subplot3: k         for the subplot3's x axis ranking k
# subplot4: k         for the subplot4's x axis ranking k
# K_cross:  k         for the k groups for cross_validation
# K_KNN:    k         for the k vote for KNN
# cross_statistic: file_path for cross_validatipn data output
# histogram : file_path  for cross_validatipn  histogram output
# heat_map: file_path  for heat_map  output
# top_k_heat_map:  k  for the top k feature of heat_map
from scipy import stats
from sklearn import svm
from sklearn import tree
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import Lasso,LassoCV,LassoLarsCV
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import SelectFromModel
from sklearn.feature_selection import RFE
import matplotlib.pyplot as plt
import os
import seaborn as sns
import numpy as np
import math
from sklearn.pipeline import Pipeline
import random
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import RobustScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier


def fLoadDataMatrix(filename):
    with open(filename, 'r') as fr:
        lines = fr.readlines()
    first_line = lines[0].strip()
    tFeature = first_line.split(',')[:-1]
    feature_count = len(tFeature)
    sample_count = len(lines) - 1
    tSample = list(range(sample_count))
    tMatrix = np.zeros((sample_count, feature_count))
    tClass = []
    for sample_index, line in enumerate(lines[1:]):
        line = line.strip()
        list_from_line = line.split(',')
        tMatrix[sample_index, :] = list_from_line[:-1]
        tClass.append(list_from_line[-1])
    ClassLabels = [int(label) for label in tClass]
    return tSample, ClassLabels, tFeature, tMatrix

def lord_conf(filename):
    config= {}
    with open(filename, encoding='utf-8') as fn:
        for line in fn:
            line = line.strip()
            if line and not line.startswith('#'):
                try:
                    key, value = line.split(':', 1)
                    config[key.strip()] = value.strip()
                except ValueError:
                    pass
    if not os.path.exists((config['Output_dir'])):
        os.makedirs(config['Output_dir'])
    return config

def preprocess(t_matrix):
    #Standardization
    scaler = StandardScaler()
    scaled_matrix_1 = scaler.fit_transform(t_matrix)
    #Normalization
    scaler = MinMaxScaler()
    scaled_matrix_2 = scaler.fit_transform(t_matrix)
    #RobustScaler
    scaler = RobustScaler()
    scaled_matrix_3 = scaler.fit_transform(scaled_matrix_2)
    return scaled_matrix_3
def classify(t_class):
    class_N = []
    class_P = []
    for i in range(len(t_class)):
        if (t_class[i] == 0):
            class_N.append(i)
        elif (t_class[i] == 1):
            class_P.append(i)
        else:
            pass
    return  class_N,class_P
def t_test(t_class,t_matrix,class_N,class_P):
    t_statistic, p_values = stats.ttest_ind(t_matrix[class_N], t_matrix[class_P])
    idx_rank_sort = np.argsort(p_values)
    return idx_rank_sort, t_statistic, p_values

def rank_lasso(t_sample, t_class, t_feature, t_matrix):

    lso = Lasso(alpha=float(conf['lasso_alpha']), random_state=5).fit(t_matrix, t_class)
    _model2 = SelectFromModel(lso, prefit=True)
    sorted_indices_lasso = _model2.get_support(indices=True)
    N_x2 = _model2.transform(t_matrix)
    fw = open(os.path.join(conf["Output_dir"], conf["lasso_feature_output"]), "w")
    for x in sorted_indices_lasso: print(t_feature[x], file=fw)
    fw.close()
    return sorted_indices_lasso


def rank_RFI(t_sample, t_class, t_feature, t_matrix,conf):
    X = np.array(t_matrix)
    y = np.array(t_class)
    forest = RandomForestClassifier(n_estimators=100, random_state=42)
    forest.fit(X, y)
    importances = forest.feature_importances_
    top_k_features = int(conf['PFI_k_feature'])
    indices = np.argsort(importances)[::-1]
    top_indices = indices[:top_k_features]
    fw = open(os.path.join(conf["Output_dir"], conf["PFI_feature_output"]), "w")
    for x in top_indices: print(t_feature[x], file=fw)
    fw.close()
    return top_indices

def print_top_k_features(sorted_indices, t_statistic, p_values,conf):
    fw =  open(os.path.join(conf["Output_dir"], conf["Top_k_features"]), "w")
    print("feature\t\t\t","p_value\t\t\t\t\t","t_statistic",file=fw)
    for i in range(int(conf['Max_feature'])):
        print(t_feature[sorted_indices[i]],'\t\t\t',p_values[sorted_indices[i]],'\t',t_statistic[sorted_indices[i]],file=fw)
    fw.close()

def print_dot_plot(t_matrix,t_feature, sorted_indices,conf,class_N,class_P):
    test_indices = [int(conf[f"subplot{i + 1}"]) - 1 for i in range(4)]  # Get indices from config
    fig = plt.figure(figsize=[13, 11], dpi=300)
    for i in range(4):
        ax = fig.add_subplot(221 + i)
        x_idx = sorted_indices[test_indices[i]]
        y_idx = sorted_indices[test_indices[i] + 1]
        ax.scatter(t_matrix[:, x_idx][class_N], t_matrix[:, y_idx][class_N], s=2, color='b', label="NEG")
        ax.scatter(t_matrix[:, x_idx][class_P], t_matrix[:, y_idx][class_P], s=2, color='r', label="POS")
        plt.legend()
        plt.xlabel(t_feature[x_idx])
        plt.ylabel(t_feature[y_idx])
        ax.set_title(f"Rank {test_indices[i] + 1} vs Rank {test_indices[i] + 2}")
    plt.savefig(os.path.join(conf['Output_dir'],conf['Dot_plot']))


def should_apply_jitter_and_alpha(t_matrix, threshold=0.5, repetition_threshold=0.3):
    repetitive_features = []
    for col in t_matrix.T:
        unique_values, counts = np.unique(col, return_counts=True)
        max_repetition = np.max(counts) / len(col)  # Calculate proportion of most frequent value
        repetitive_features.append(max_repetition >= repetition_threshold)
    proportion_repetitive_features = np.mean(repetitive_features)
    return proportion_repetitive_features > threshold

def print_dot_plot_jitter_and_alpha(t_matrix,t_feature, sorted_indices,conf,class_N,class_P):
    test_indices = [int(conf[f"subplot{i + 1}"]) - 1 for i in range(4)]
    fig = plt.figure(figsize=[13, 11], dpi=300)
    for i in range(4):
        ax = fig.add_subplot(221 + i)
        x_idx = sorted_indices[test_indices[i]]
        y_idx = sorted_indices[test_indices[i] + 1]
        apply_jitter_alpha = should_apply_jitter_and_alpha(t_matrix[:, [x_idx, y_idx]])
        if apply_jitter_alpha:
            jitter_scale_x = 0.02 * ( np.nanmax(t_matrix[:, x_idx]) - np.nanmin(t_matrix[:, x_idx]) )
            jitter_scale_y = 0.02 * ( np.nanmax(t_matrix[:, y_idx]) - np.nanmin(t_matrix[:, y_idx]) )
            x_jittered = np.random.normal(t_matrix[:, x_idx], jitter_scale_x)
            y_jittered = np.random.normal(t_matrix[:, y_idx], jitter_scale_y)
            ax.scatter(x_jittered[class_N], y_jittered[class_N], s=2, color='b', label="NEG", alpha=0.2)
            ax.scatter(x_jittered[class_P], y_jittered[class_P], s=2, color='r', label="POS", alpha=0.2)
            ax.set_title(f"Rank {test_indices[i] + 1} vs Rank {test_indices[i] + 2} (jittered)")
        else:
            ax.scatter(t_matrix[:, x_idx][class_N], t_matrix[:, y_idx][class_N], s=2, color='b', label="NEG")
            ax.scatter(t_matrix[:, x_idx][class_P], t_matrix[:, y_idx][class_P], s=2, color='r', label="POS")
            ax.set_title(f"Rank {test_indices[i] + 1} vs Rank {test_indices[i] + 2}")
        plt.xlabel(t_feature[x_idx])
        plt.ylabel(t_feature[y_idx])
        plt.legend()

    plt.savefig(os.path.join(conf['Output_dir'],conf['Dot_plot_jitter_and_alpha']))

def KNN(test_x,train_x, train_y, k_KNN):
    matri_size = train_x.shape[0]
    diff = np.tile(test_x, (matri_size, 1)) - train_x
    sq_diff= diff ** 2
    sq_dist = sq_diff.sum(axis = 1)
    distances = sq_dist ** 0.5
    sorted_dist_indices = distances.argsort()
    class_count = {}
    for i in range(k_KNN) :
        vote_i_label = train_y[sorted_dist_indices[i+1]]
        class_count[vote_i_label] = class_count.get(vote_i_label, 0) + 1
    sorted_class_count = sorted(class_count.items(), key = lambda d: d[1], reverse = True)
    return  sorted_class_count[0][0]

def train_NB(train_x,train_y):
    num_sample = len(train_x)
    class_P = []
    class_N = []
    for i in range(num_sample):
        if (train_y[i] == 1):
            class_P.append(i)
        elif (train_y[i] == 0):
            class_N.append(i)
        else:
            pass
    mat_P = train_x[class_P][:]
    mat_N = train_x[class_N][:]
    p_abusive = len(class_P) / float(num_sample)
    P_avg = np.mean(mat_P, 0)
    N_avg = np.mean(mat_N, 0)
    P_delta = np.var(mat_P, 0)
    N_delta = np.var(mat_N, 0)
    return P_avg, N_avg, P_delta, N_delta, p_abusive

def classify_NB(test_x, P_avg, N_avg, P_delta, N_delta, p_abusive):
    def normal_distribution(mu, sig, x):
        return math.exp(-math.pow(x - mu, 2) / (2 * sig)) / (math.sqrt(2 * math.pi) * math.sqrt(sig))
    numFeature = len(test_x)
    p1 = p_abusive
    p0 = 1 - p_abusive
    for i in range(numFeature):
        p1 *= normal_distribution(P_avg[i], P_delta[i], test_x[i])
        p0 *= normal_distribution(N_avg[i], N_delta[i], test_x[i])
    return  p1 > p0

def evaluate(predict, origin):
    confusion_mat = np.zeros((2, 3))
    for i in range(len(origin)):
        confusion_mat[origin[i]][2] += 1
        confusion_mat[origin[i]][predict[i]] += 1
    result = list(range(5))
    TP = confusion_mat[1][1]    #True Positive
    TN = confusion_mat[0][0]
    FP = confusion_mat[1][0]    #False
    FN = confusion_mat[0][1]
    P = confusion_mat[1][2]
    N = confusion_mat[0][2]
    result[0] = TP / P               # pos acc
    result[1] = TN / N               # neg acc
    result[2] = (TP + TN) / (P + N)  # acc
    result[3] = (result[0] + result[1]) / 2
    x1 = (TP * TN - FP * FN)
    x2 = (TP + FP) * (TP + FN) * (TN + FP) * (TN + FN)
    if (x2 == 0):                   # MCC
        result[4] = 0               # [ -1 ,  1 ]
    else:                           # [bad ,good]
        result[4] = x1 / math.sqrt(x2)
    return result

def cross_validation_one_round(t_data,t_class,group_N,group_P,conf,fw):
    k = int(conf['K_cross'])
    k_KNN = int(conf['K_KNN'])
    result = np.zeros((5, 5))

    for cnt in range(k):
        print("fold"+str(cnt),file=fw)
        test_group = group_N[cnt] + group_P[cnt]
        train_group = []
        for i in range(k):
            if (i != cnt):
                train_group = train_group + group_N[i] + group_P[i]
            else:
                pass
        train_x = t_data[train_group, :]
        train_y = np.asarray([t_class[x] for x in train_group])
        test_x = t_data[test_group, :]
        test_y = np.asarray([t_class[x] for x in test_group])

        #######################
        #       KNN           #
        #######################
        predict_KNN = np.zeros((len(test_y)), dtype=np.int64)
        for i in range(len(test_y)):
             predict_KNN[i] = KNN(test_x[i], train_x, train_y, k_KNN)

        #######################
        #        NB           #
        #######################
        predict_NB = np.zeros((len(test_y)), dtype=np.int64)
        P_avg, N_avg, P_delta, N_delta, p_abusive = train_NB(train_x,train_y)
        for i in range(len(test_y)):
            predict_NB[i] = classify_NB(test_x[i], P_avg, N_avg, P_delta, N_delta, p_abusive)

        #######################
        #        SVC          #
        #######################
        pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('svc', svm.SVC())
        ])
        param_grid = {
            'svc__C': [0.1, 1, 3,  5 , 10, 100],
            'svc__kernel': ['linear', 'poly', 'rbf', 'sigmoid'],
            'svc__gamma': ['scale', 'auto']
        }
        grid_search = GridSearchCV(pipeline, param_grid, cv=5, n_jobs=-1)
        grid_search.fit(train_x, train_y)
        print("Best parameters for SVC found:", grid_search.best_params_,file=fw)
        best_SVC_model = grid_search.best_estimator_
        predict_SVC = best_SVC_model.predict(test_x)
        #######################
        #        dtree        #
        #######################
        param_grid = {
            'criterion': ['gini', 'entropy', 'log_loss'],
            'max_depth': [None, 5, 10, 15],
            'min_samples_split': [2, 5, 10]
        }
        model_dtree = DecisionTreeClassifier()
        grid_search = GridSearchCV(model_dtree, param_grid, cv=5, n_jobs=-1)
        grid_search.fit(train_x, train_y)
        print("Best parameters for dtree found:", grid_search.best_params_,file=fw)
        best_model = grid_search.best_estimator_
        predict_dtree = best_model.predict(test_x)
        #######################
        #        Lasso        #
        #######################
        model_Lasso = LassoLarsCV(cv=5)
        model_Lasso.fit(train_x, train_y)
        predict_Lasso = np.zeros((len(test_y)), dtype=np.int64)
        predictarr = model_Lasso.predict(test_x)
        for i in range(len(test_y)):
            predict_Lasso[i] = (predictarr[i] > 0.5)
        ev1 = evaluate(predict_KNN, test_y)
        ev2 = evaluate(predict_NB, test_y)
        ev3 = evaluate(predict_SVC, test_y)
        ev4 = evaluate(predict_dtree, test_y)
        ev5 = evaluate(predict_Lasso, test_y)
        result += np.asarray([ev1, ev2, ev3, ev4, ev5])
    return  result / k
def cross_validation(sorted_indices,t_class,t_matrix,conf,class_N,class_P,):
    fw = open(os.path.join(conf["Output_dir"], conf["cross_statistic"]), "w")
    k=int(conf['K_cross'])
    random.shuffle(class_N)
    random.shuffle(class_P)
    group_N = []
    group_P = []
    for i in range(k):
        group_N.append(class_N[i::k])
        group_P.append(class_P[i::k])
    method= conf['Choose_top_k']
    result=[]
    if method=='t_test':
        test_indices = [int(conf[f"subplot{i + 1}"]) for i in range(3)]
        for i in range(3):
            t_data = t_matrix[:,sorted_indices[:test_indices[i]]]
            result.append( cross_validation_one_round(t_data,t_class,group_N, group_P,conf,fw) )
        t_data = t_matrix[:,sorted_indices[-5:]]
        result.append(cross_validation_one_round(t_data, t_class, group_N, group_P, conf,fw))
    else:
        t_data = t_matrix[:, sorted_indices]
        result.append(cross_validation_one_round(t_data, t_class, group_N, group_P, conf,fw))


    print(result,file=fw)
    fw.close()
    return  result

def print_histograms(result, conf):
    method = conf['Choose_top_k']
    if method == 't_test':
        test_indices = [(conf[f"subplot{i + 1}"]) for i in range(4)]
        fig = plt.figure(figsize=(12, 7), dpi = 120)
        X_base = np.asarray([1, 7, 13, 19, 25])
        title = [ "Top-"+ test_indices[0], "Top-"+ test_indices[1], "Top-"+ test_indices[2] ,"Bottom-5"  ]
        for i in range(4):
            errors = (np.array(result)).std(axis=0)
            label = ["KNN", "NBayes", "SVM", "DTree", "Lasso"]
            colors = ['r', 'g', 'b', 'orange', 'purple']
            plt.subplot(221 + i)
            if(i==3):
                plt.ylim(-0.3, 1.1)
            else:
                plt.ylim(-0.1, 1.1)
            plt.xlim(0, 35)
            for j in range(5):
                plt.bar(X_base+j,
                    result[i][j],
                    1,
                    label=label[j],
                    color=colors[j]
                )
                plt.errorbar(
                    X_base + j,
                    result[i][j],
                    yerr=errors[j],
                    fmt='none',
                    capsize=3,
                    color=colors[j],
                    ecolor=colors[j]
                )
            plt.xticks(X_base+2, ('Sn', 'Sp', 'Acc', 'Avc', 'MCC'))
            plt.title(title[i])
            plt.legend(loc="upper right", fontsize = "small" )
    else :
        if method=='lasso':
            title= ["Top feature under lasso with alpha:"+conf['lasso_alpha']]
        else:
            title = ["Top-"+conf['PFI_k_feature']+"feature under RFI"]
        fig = plt.figure(figsize=(12, 7), dpi=120)
        X_base = np.asarray([1, 7, 13, 19, 25])
        errors = (np.array(result)).std(axis=0)
        label = ["KNN", "NBayes", "SVM", "DTree", "Lasso"]
        colors = ['r', 'g', 'b', 'orange', 'purple']
        i=0
        plt.subplot(111 + i)
        if (i == 3):
            plt.ylim(-0.3, 1.1)
        else:
            plt.ylim(-0.1, 1.1)
        plt.xlim(0, 35)
        for j in range(5):
            plt.bar(X_base + j,
                    result[i][j],
                    1,
                    label=label[j],
                    color=colors[j]
                    )
        plt.xticks(X_base + 2, ('Sn', 'Sp', 'Acc', 'Avc', 'MCC'))
        plt.title(title[i])
        plt.legend(loc="upper right", fontsize="small")
    plt.savefig(os.path.join(conf["Output_dir"], conf["histogram"]))

def print_heat_map(t_class, t_matrix, sorted_indices, t_feature, conf):
    method= conf['Choose_top_k']
    if method == 't_test':
        k = int(conf['top_k_heat_map'])
        feature= [t_feature[i] for i in sorted_indices[:k] ]
        matrix = t_matrix[:,sorted_indices[:k]]
        color = {0: 'b', 1: 'r'}
        color_label = [color[i] for i in t_class]
        sns.set()
        hm = sns.clustermap(matrix.T, col_colors=color_label, yticklabels=feature)
        ax = hm.ax_heatmap
        ax.set_yticklabels(ax.get_yticklabels(), rotation=0)
        ax.set_xticklabels([])
        plt.title("Top-" + str(k) + " features " + method, y=0.95)
        plt.subplots_adjust(left=0.005)
        plt.savefig(os.path.join(conf["Output_dir"], conf["heat_map"]))
    elif method == 'lasso':
        feature = [t_feature[i] for i in sorted_indices]
        matrix = t_matrix[:, sorted_indices]
        color = {0: 'b', 1: 'r'}
        color_label = [color[i] for i in t_class]
        sns.set()
        hm = sns.clustermap(matrix.T, col_colors=color_label, yticklabels=feature)
        ax = hm.ax_heatmap
        ax.set_yticklabels(ax.get_yticklabels(), rotation=0)
        ax.set_xticklabels([])
        plt.title("Top features under+" + method, y=0.95)
        plt.subplots_adjust(left=0.005)
        plt.savefig(os.path.join(conf["Output_dir"], conf["lasso_feature_heat_map"]))
    else:
        k = int(conf['PFI_k_feature'])
        feature = [t_feature[i] for i in sorted_indices]
        matrix = t_matrix[:, sorted_indices]
        color = {0: 'b', 1: 'r'}
        color_label = [color[i] for i in t_class]
        sns.set()
        hm = sns.clustermap(matrix.T, col_colors=color_label, yticklabels=feature)
        ax = hm.ax_heatmap
        ax.set_yticklabels(ax.get_yticklabels(), rotation=0)
        ax.set_xticklabels([])
        plt.title("Top- "+str(k)+" features under+" + method, y=0.95)
        plt.subplots_adjust(left=0.005)
        plt.savefig(os.path.join(conf["Output_dir"], conf["RFI_feature_heat_map"]))

def perform_t_test(t_sample, t_class, t_feature, t_matrix,conf,class_N, class_P):
    sorted_indices, t_statistic, p_values = t_test(t_class, t_matrix, class_N, class_P)

    print_top_k_features(sorted_indices, t_statistic, p_values, conf)

    print_dot_plot(t_matrix, t_feature, sorted_indices, conf, class_N, class_P)

    print_dot_plot_jitter_and_alpha(t_matrix, t_feature, sorted_indices, conf, class_N, class_P)

    result = cross_validation(sorted_indices, t_class, t_matrix, conf, class_N, class_P)

    print_histograms(result, conf)

    print_heat_map(t_class, t_matrix, sorted_indices, t_feature, conf)

def perform_lasso(t_sample, t_class, t_feature, t_matrix,conf,class_N, class_P):
    sorted_indices_lasso = rank_lasso(t_sample, t_class, t_feature, t_matrix)

    result = cross_validation(sorted_indices_lasso, t_class, t_matrix, conf, class_N, class_P)

    print_histograms(result, conf)

    print_heat_map(t_class, t_matrix, sorted_indices_lasso, t_feature, conf)

def perform_RFI(t_sample, t_class, t_feature, t_matrix,conf,class_N, class_P):
    sorted_indices_RFI= rank_RFI(t_sample, t_class, t_feature, t_matrix, conf)

    result = cross_validation(sorted_indices_RFI, t_class, t_matrix, conf, class_N, class_P)

    print_histograms(result, conf)

    #print_heat_map(t_class, t_matrix, sorted_indices_RFI, t_feature, conf)
try:
    conf = lord_conf('/home/violesa/PycharmProjects/pythonProject4/pbc.conf')
except FileNotFoundError:
    print("Error: Configuration file not found.")
    exit(1)

try:
    t_sample, t_class, t_feature, t_matrix = fLoadDataMatrix('/home/violesa/Downloads/archive/AIDS_Classification.csv')
except FileNotFoundError:
    print("Error: Input file not found.")
    exit(1)

t_matrix=preprocess(t_matrix)

class_N, class_P = classify(t_class)

Choose_top_k = conf['Choose_top_k']

if Choose_top_k=='t_test':
    perform_t_test(t_sample, t_class, t_feature, t_matrix,conf,class_N, class_P)
elif Choose_top_k == 'lasso':
    perform_lasso(t_sample, t_class, t_feature, t_matrix,conf,class_N, class_P)
elif Choose_top_k == 'RFI':
    perform_RFI(t_sample, t_class, t_feature, t_matrix,conf,class_N, class_P)
else: print("wrong Choose_top_k arg")
#超参数调优： 对于 SVM、Decision Tree 和 LassoLarsCV 等模型，可以尝试调整超参数，以提高模型性能。
