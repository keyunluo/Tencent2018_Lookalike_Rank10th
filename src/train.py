# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import tensorflow as tf
import utils
import nffm
import os
os.environ["CUDA_DEVICE_ORDER"] = 'PCI_BUS_ID'
os.environ["CUDA_VISIBLE_DEVICES"] = '0'


def create_hparams():
    return tf.contrib.training.HParams(
        k=8,
        batch_size=4096,
        optimizer="adam",
        learning_rate=0.0002,
        num_display_steps=100,
        num_eval_steps=1000,
        l2=0.000002,
        hidden_size=[256, 128],
        evl_batch_size=4096,
        all_process=1,
        idx=0,
        epoch=int(44628906//4096),
        mode='train',
        data_path='dataset/',
        sub_name='nffm',
        single_features=['aid', 'advertiserId', 'campaignId', 'creativeId', 'creativeSize', 'adCategoryId', 'productId', 'productType', 'age', 'gender',
                         'education', 'consumptionAbility', 'LBS', 'carrier', 'house', 'uid_aid_nunique', 'aid_uid_nunique', 'campaignId_aid_nunique', 'uid_convert'],
        mutil_features=['interest1', 'interest2', 'interest3', 'interest4', 'interest5', 'kw1', 'kw2', 'kw3', 'topic1', 'topic2', 'topic3', 'appIdAction',
                        'appIdInstall', 'marriageStatus', 'ct', 'os', 'pos_aid', 'neg_aid']
    )


hparams = create_hparams()
hparams.path = '../model/'

hparams.aid = ['aid', 'advertiserId', 'campaignId', 'creativeId', 'creativeSize', 'adCategoryId', 'productId', 'productType', 'pos_aid',
               'neg_aid', 'aid_uid_nunique', 'campaignId_aid_nunique']
hparams.user = ['age', 'gender', 'education', 'consumptionAbility', 'LBS', 'carrier', 'house', 'interest1', 'interest2', 'interest3', 'interest4', 'interest5',
                'kw1', 'kw2', 'kw3', 'topic1', 'topic2', 'topic3', 'appIdAction', 'appIdInstall', 'marriageStatus', 'ct', 'os', 'uid_aid_nunique', 'uid_convert']
hparams.num_features = []
preds = nffm.train(hparams)


test2_df = pd.read_csv('input/test2.csv')
test2_df['score'] = preds
test2_df['score'] = test2_df['score'].apply(lambda x: round(x, 4))
test2_df[['aid', 'uid', 'score']].to_csv(
    '../submission_'+str(hparams.sub_name)+'.csv', index=False)
