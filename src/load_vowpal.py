# -*- coding: utf-8 -*-

from csv import DictWriter
from tqdm import tqdm
import mmap


def get_file_lines(file_path):
    '''
    获取文件行数
    '''
    fp = open(file_path, "r+")
    buf = mmap.mmap(fp.fileno(), 0)
    lines = 0
    while buf.readline():
        lines += 1
    return lines


def process_vowpal(file_path, out_path):
    '''
    处理用户特征文件
    '''
    headers = ['uid', 'age', 'gender', 'marriageStatus', 'education', 'consumptionAbility', 'LBS', 'interest1', 'interest2', 'interest3',
               'interest4', 'interest5', 'kw1', 'kw2', 'kw3',  'topic1', 'topic2', 'topic3', 'appIdInstall', 'appIdAction', 'ct', 'os', 'carrier', 'house']
    fo = open(out_path, 'wt')
    writer = DictWriter(fo, fieldnames=headers, lineterminator='\n')
    writer.writeheader()

    with open(file_path, 'rt') as f:
        for line in tqdm(f, total=get_file_lines(file_path)):
            feature_groups = line.strip().split('|')
            fea_dict = {}
            for feas in feature_groups:
                feas_split = feas.split(' ')
                fea_dict[feas_split[0]] = ' '.join(feas_split[1:])
            writer.writerow(fea_dict)
    fo.close()


if __name__ == '__main__':
    file_path = 'input/userFeature.data'
    out_path = 'input/userFeature.csv'
    process_vowpal(file_path, out_path)
