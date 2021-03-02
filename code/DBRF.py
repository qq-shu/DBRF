import os
import json

import joblib
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN, OPTICS
import pandas as pd
from imblearn.over_sampling import BorderlineSMOTE
from sklearn.ensemble import RandomForestClassifier
import numpy as np

from k_nearest_neighbors import get_neighbors


def get_training_c(data, k, feature_list, min_label):
    rfData = data.to_numpy()
    max_data = list()
    min_data = list()
    training_c = list()

    for row in rfData:
        curr_label = row[-1]
        if curr_label == min_label:
            min_data.append(row)
        else:
            max_data.append(row)

    # find the potential problem areas affecting the minority instances
    for row_i in min_data:
        # update critical dataset
        training_c.append(row_i)
        # find k nearest neighbors for each minority instance in the data set
        neighbors = get_neighbors(row_i, max_data, 10)
        # add unique neighbors to the critical data set
        for row_j in neighbors:
            if not any((row_j == x).all() for x in training_c):
                training_c.append(row_j)

    df_training_c = pd.DataFrame(training_c, columns=feature_list)
    return df_training_c


def Borderline_DBSCAN(train_data, label, eps=20.1, min_samples=5):
    label_index = 0
    if label == 'c':
        label_index = 1
    if label == 'b':
        label_index = 0

    print(train_data['label'].value_counts())
    boSMOTE = BorderlineSMOTE(kind='borderline-1')
    x, y = boSMOTE.fit_resample(train_data.iloc[:, :-1], train_data.iloc[:, -1])

    # print(boSMOTE.sample)
    BMG_sample = boSMOTE.sample[label_index][1]
    BMG_sample = pd.DataFrame(BMG_sample, columns=train_data.columns.values.tolist()[:-1])
    BMG_sample['label'] = label


    max_sample = []
    min_sample = []
    # print(train_data.shape[0])
    for temp in range(train_data.shape[0]):
        if train_data.iloc[temp, -1] == label:
            min_sample.append(train_data.iloc[temp, :].values)
        else:
            max_sample.append(train_data.iloc[temp, :].values)

    max_sample = pd.DataFrame(max_sample, columns=train_data.columns.values.tolist())
    min_sample = pd.DataFrame(min_sample, columns=train_data.columns.values.tolist())
    mergeSample = pd.concat([max_sample, BMG_sample], ignore_index=False)
    # print(min_sample.shape[0])
    # print(max_sample.shape[0])
    # print("**9**")
    # print(mergeSample.shape[0])
    dbsc = DBSCAN(eps=eps, min_samples=min_samples).danger_fit(X=mergeSample, danger_sample=BMG_sample)
    array_neighborhoods = dbsc.neighborhoods
    neighborhoods_index = []
    array_n_neighbors = dbsc.n_neighbors
    for temp in range(len(array_n_neighbors)):
        if array_n_neighbors[temp] >= 5:
            for i in range(array_n_neighbors[temp]):
                neighborhoods_index.append(array_neighborhoods[temp][i])
    new_sample_index = list(set(neighborhoods_index))
    num_sample = BMG_sample.shape[0]
    # print(array_neighborhoods)
    # print(len(new_sample_index))
    # print(train_data.shape[0])
    return min_sample, mergeSample, new_sample_index, num_sample


def get_mod_training_c(df, labels_list, min_list, classif_num):
    df = df[df['label'].isin(labels_list)].sample(frac=1).reset_index(drop=True)
    train_count = int(0.75 * len(df))
    train_data = df.loc[:train_count, :]
    test_data = df.loc[train_count:, :]

    training_c = get_training_c(data=train_data, k=10, feature_list=train_data.columns.values.tolist(),
                                min_label=min_list)
    if classif_num == 2:
        BMG_min_sample, BMG_merge_sample, BMG_new_sample_index, BMG_num_sample = Borderline_DBSCAN(train_data,
                                                                              label=min_list)
        training_c = get_training_c(data=train_data, k=10, feature_list=train_data.columns.values.tolist(),
                                    min_label=min_list)
        BMG_set = set(range(BMG_num_sample))
        set_BMG = set(BMG_new_sample_index) - BMG_set
        train_c_sample_index = list(set_BMG)

        a = []
        for j in train_c_sample_index:
            if train_data.iloc[j, -1] != min_list:
                a.append(j)
        new_sample = train_data.iloc[a, :]
        # min_sample = pd.DataFrame()
        mod_training_c = pd.concat([new_sample, BMG_min_sample], ignore_index=True)
        return train_data, test_data, mod_training_c, training_c, new_sample

    if classif_num == 3:
        training_c = get_training_c(data=train_data, k=10, feature_list=train_data.columns.values.tolist(),
                                    min_label=min_list[0])
        BMG_min_sample, BMG_merge_sample, BMG_new_sample_index, BMG_num_sample = Borderline_DBSCAN(train_data,
                                                                                                   label=min_list[0])
        CRA_min_sample, CRA_merge_sample, CRA_new_sample_index, CRA_num_sample = Borderline_DBSCAN(train_data,
                                                                                                   label=min_list[1])

        BMG_set = set(range(BMG_num_sample))
        CRA_set = set(range(CRA_num_sample))

        set_BMG = set(BMG_new_sample_index) - BMG_set
        set_CRA = set(CRA_new_sample_index) - CRA_set
        train_c_sample_index = list(set(list(set_BMG) + list(set_CRA)))

        a = []

        for j in train_c_sample_index:
            # print(str(j) + "      " + str(train_data.shape[0]))
            if train_data.iloc[j, -1] != min_list[0] and train_data.iloc[j, -1] != min_list[1]:
                a.append(j)
        new_sample = train_data.iloc[a, :]
        min_sample = pd.concat([BMG_min_sample, CRA_min_sample], ignore_index=True)
        mod_training_c = pd.concat([new_sample, min_sample], ignore_index=True)
        return train_data, test_data, mod_training_c, training_c, new_sample
        # return train_data, test_data, mod_training_c

# df = pd.read_csv("data/Full_Dataset-Dmax-TTT.csv")
# label_list = ['BMG', 'CRA', 'RMG']
# min_list = ['BMG', 'CRA']
# class_num = 3

df = pd.read_csv("data/haberman.data")
# df = df.iloc[:, 1:]
# df = df.loc[:, ['Atr-0', 'Atr-4', 'Atr-5', 'Atr-6', 'Atr-7', 'Atr-8', 'Atr-9', 'Atr-10', 'Atr-11', 'Atr-12', 'Atr-13', 'Atr-14', 'Atr-15', 'Atr-16', 'Atr-17', 'Atr-18', 'Atr-19', 'Atr-20', 'Atr-21', 'Atr-22', 'Atr-23', 'Atr-24', 'Atr-25', 'Atr-26', 'Atr-27', 'Atr-28', 'Atr-29', 'Atr-30', 'Atr-31', 'Atr-32', 'Atr-33', 'Atr-34', 'Atr-35', 'Atr-36', 'Atr-37', 'Atr-38', 'Atr-39', 'Atr-40', 'label']]
label_list = [2, 1]
min_list = 2
class_num = 2
argList = []
for i in range(10):
    print(i)
    train_data, test_data, mod_training_c, training_c, new_sample = get_mod_training_c(df=df, labels_list=label_list, min_list=min_list, classif_num=class_num)
    # data_path = 'data/BorderlineDBSCAN/' + str(i)
    data_path = 'data/BorderlineDBSCAN/haberman_true/' + str(i)

    train_path = data_path + '/train.csv'
    train_data.to_csv(train_path)
    test_path = data_path + '/test.csv'
    test_data.to_csv(test_path)
    training_c_path = data_path + '/training_c.csv'
    training_c.to_csv(training_c_path)
    mod_training_c_path = data_path + '/mod_training_c.csv'
    mod_training_c.to_csv(mod_training_c_path)
    new_sample_path = data_path + "/new_sample.csv"
    new_sample.to_csv(new_sample_path)

    train_x = train_data.iloc[:, :-1]
    train_y = train_data.iloc[:, -1]
    test_x = test_data.iloc[:, :-1]
    test_y = test_data.iloc[:, -1]
    mod_training_c_x = mod_training_c.iloc[:, :-1]
    mod_training_c_y = mod_training_c.iloc[:, -1]
    training_c_x = training_c.iloc[:, :-1]
    training_c_y = training_c.iloc[:, -1]
    p_critical_area_ratio = 0.1
    rfSize = 100
    rootPath = os.getcwd()
    model_path = data_path

    # 50棵原始树
    rf1_size = int(rfSize * (1 - p_critical_area_ratio))
    rf1 = RandomForestClassifier(n_estimators=rf1_size)
    rf1.fit(train_x, train_y)
    rf1_path = model_path + '/rf1.m'
    joblib.dump(rf1, rf1_path)

    # 50棵关键区域树
    rf2_size = int(rfSize * p_critical_area_ratio)
    rf2 = RandomForestClassifier(n_estimators=rf2_size)
    rf2.fit(mod_training_c_x, mod_training_c_y)
    rf2_path = model_path + '/rf2.m'
    joblib.dump(rf2, rf2_path)

    # 原始森林
    RF = RandomForestClassifier(n_estimators=rfSize)
    RF.fit(train_x, train_y)
    RF_path = model_path + '/RF.m'
    joblib.dump(RF, RF_path)

    # BRAF
    rf3 = RandomForestClassifier(n_estimators=rf2_size)
    rf3.fit(training_c_x, training_c_y)
    rf3_path = model_path + '/rf3.m'
    joblib.dump(rf3, rf3_path)

    RF1 = RandomForestClassifier(n_estimators=rfSize)
    Gobaltree = rf1.estimators_ + rf3.estimators_
    RF1.estimators_ = Gobaltree
    RF1.classes_ = rf1.classes_
    RF1.n_classes_ = rf1.n_classes_
    RF1.n_outputs_ = rf1.n_outputs_
    RF1_path = model_path + '/braf.m'
    joblib.dump(RF1, RF1_path)

    # DBRF
    RF2 = RandomForestClassifier(n_estimators=rfSize)
    mod_Gobaltree = rf1.estimators_ + rf2.estimators_
    RF2.estimators_ = mod_Gobaltree
    RF2.classes_ = rf2.classes_
    RF2.n_classes_ = rf2.n_classes_
    RF2.n_outputs_ = rf2.n_outputs_
    RF2_path = model_path + '/borderlindbscan.m'
    joblib.dump(RF2, RF2_path)

    from sklearn import metrics

    hunxiao1 = metrics.confusion_matrix(test_y, RF.predict(test_x))
    hunxiao2 = metrics.confusion_matrix(test_y, RF1.predict(test_x))
    hunxiao3 = metrics.confusion_matrix(test_y, RF2.predict(test_x))
    file_path1 = data_path + '/hunxiao1.json'
    file_path2 = data_path + "/hunxiao2.json"
    file_path3 = data_path + "/hunxiao3.json"
    with open(file_path1, 'w') as jsonfile:
        hunxiao1 = {'hunxioa': str(hunxiao1)}
        jsonfile.write(json.dumps(hunxiao1) + "\n")
    with open(file_path2, 'w') as jsonfile:
        hunxiao2 = {'hunxioa': str(hunxiao2)}
        jsonfile.write(json.dumps(hunxiao2) + "\n")
    with open(file_path3, 'w') as jsonfile:
        hunxiao3 = {'hunxioa': str(hunxiao3)}
        jsonfile.write(json.dumps(hunxiao3) + "\n")
    print("end" + str(i))



