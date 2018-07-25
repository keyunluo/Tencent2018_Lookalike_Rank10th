# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import random
import pickle
import gc
import os
from collections import Counter, OrderedDict
from sklearn.model_selection import train_test_split


threshold = 1000
random.seed(2018)


def pre_data():
    user_feature = pd.read_csv('input/userFeature.csv')
    ad_Feature = pd.read_csv('input/adFeature.csv')

    train_df = pd.read_csv('input/train.csv')
    test_df = pd.read_csv('input/test2.csv')

    train_df = pd.merge(train_df, ad_Feature, on='aid', how='left')
    test_df = pd.merge(test_df, ad_Feature, on='aid', how='left')
    train_df = pd.merge(train_df, user_feature, on='uid', how='left')
    test_df = pd.merge(test_df, user_feature, on='uid', how='left')

    del user_feature
    gc.collect()

    # ['pos_aid', 'neg_aid']
    train_aid_fea = pd.read_csv(
        'dataset/train_neg_pos_aid.csv', usecols=['pos_aid', 'neg_aid'])
    test2_aid_fea = pd.read_csv(
        'dataset/test2_neg_pos_aid.csv', usecols=['pos_aid', 'neg_aid'])

    # ['campaignId_aid_nunique', 'uid_aid_nunique', 'aid_uid_nunique', 'uid_count', 'uid_convert']
    train_statistic_fea = pd.read_csv('dataset/train_uid_aid_bin.csv')
    test2_statistic_fea = pd.read_csv('dataset/test2_uid_aid_bin.csv')

    train_df = pd.concat(
        [train_df, train_aid_fea, train_statistic_fea], axis=1)
    test_df = pd.concat([test_df, test2_aid_fea, test2_statistic_fea], axis=1)

    train_df = train_df.fillna('0')
    test_df = test_df.fillna('0')

    gc.collect()

    train_df.loc[train_df['label'] == -1, 'label'] = 0
    test_df['label'] = -1

    train_df, dev_df = train_test_split(
        train_df, test_size=0.1, random_state=2018)

    np.save('dataset/train_df.index', np.array(train_df.index))
    np.save('dataset/dev_df.index', np.array(dev_df.index))

    return train_df, dev_df, test_df


def output_label(train_df, dev_df, test_df):
    with open('dataset/dev/label', 'w') as f:
        for i in list(dev_df['label']):
            f.write(str(i)+'\n')
    with open('dataset/test/label', 'w') as f:
        for i in list(test_df['label']):
            f.write(str(i)+'\n')
    with open('dataset/train/label', 'w') as f:
        for i in list(train_df['label']):
            f.write(str(i)+'\n')


def single_features(train_df, dev_df, test_df, word2index):
    single_ids_features = ['aid', 'advertiserId', 'campaignId', 'creativeId', 'creativeSize', 'adCategoryId', 'productId', 'productType', 'age',
                           'gender', 'education', 'consumptionAbility', 'LBS', 'carrier', 'house',
                           'campaignId_aid_nunique', 'uid_aid_nunique', 'aid_uid_nunique', 'uid_count', 'uid_convert']

    for s in single_ids_features:
        print(s)
        cont = {}

        with open('dataset/train/'+str(s), 'w') as f:
            for line in list(train_df[s].values):
                f.write(str(line)+'\n')
                if str(line) not in cont:
                    cont[str(line)] = 0
                cont[str(line)] += 1

        with open('dataset/dev/'+str(s), 'w') as f:
            for line in list(dev_df[s].values):
                f.write(str(line)+'\n')

        with open('dataset/test/'+str(s), 'w') as f:
            for line in list(test_df[s].values):
                f.write(str(line)+'\n')
        index = []
        for k in cont:
            if s not in ['campaignId_aid_nunique', 'uid_aid_nunique', 'aid_uid_nunique', 'uid_count', 'uid_convert']:
                if cont[k] >= threshold:
                    index.append(k)
            else:
                index.append(k)
        word2index[s] = {}
        for idx, val in enumerate(index):
            word2index[s][val] = idx+2
        print(s+' done!')


def mutil_ids(train_df, dev_df, test_df, word2index):
    features_mutil = ['interest1', 'interest2', 'interest3', 'interest4', 'interest5', 'kw1', 'kw2', 'kw3', 'topic1',
                      'topic2', 'topic3', 'appIdAction', 'appIdInstall', 'marriageStatus', 'ct', 'os', 'pos_aid', 'neg_aid']
    for s in features_mutil:
        print(s)
        cont = {}
        with open('dataset/train/'+str(s), 'w') as f:
            for lines in list(train_df[s].values):
                lines = str(lines)
                f.write(lines+'\n')
                for line in lines.split():
                    if line not in cont:
                        cont[line] = 0
                    cont[line] += 1

        with open('dataset/dev/'+str(s), 'w') as f:
            for line in list(dev_df[s].values):
                f.write(str(line)+'\n')

        with open('dataset/test/'+str(s), 'w') as f:
            for line in list(test_df[s].values):
                f.write(str(line)+'\n')
        index = []
        for k in cont:
            if s not in ['pos_aid', 'neg_aid']:
                if cont[k] >= threshold:
                    index.append(k)
            else:
                index.append(k)
        word2index[s] = {}
        for idx, val in enumerate(index):
            word2index[s][val] = idx+2
        print(s+' done!')


if __name__ == '__main__':
    if os.path.exists('dataset/dic.pkl'):
        word2index = pickle.load(open('dataset/dic.pkl', 'rb'))
    else:
        word2index = {}
    print("Loading Data...")
    train_df, dev_df, test_df = pre_data()

    print("Output Label...")
    output_label(train_df, dev_df, test_df)

    print("Processing Single Feature...")
    single_features(train_df, dev_df, test_df, word2index)
    pickle.dump(word2index, open('dataset/dic.pkl', 'wb'))

    print("Processing Multiple Feature...")
    mutil_ids(train_df, dev_df, test_df, word2index)
    pickle.dump(word2index, open('dataset/dic.pkl', 'wb'))
