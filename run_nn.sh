#!/bin/bash  
printf '========== Generate NN Result ==========\n'
CURDIR="`pwd`"/"`dirname $0`"
printf "\nCurrent Path：$CURDIR \n"

train_file=${CURDIR}"/src/input/train.csv"
test1_file=${CURDIR}"/src/input/test1.csv"
test2_file=${CURDIR}"/src/input/test2.csv"
ad_feature=${CURDIR}"/src/input/adFeature.csv"
user_file=${CURDIR}"/src/input/userFeature.data"
user_feature=${CURDIR}"/src/input/userFeature.csv"
dataset_dir=${CURDIR}"/src/dataset/"
train_dir=${CURDIR}"/src/dataset/train/"
valid_dir=${CURDIR}"/src/dataset/dev/"
test2_dir=${CURDIR}"/src/dataset/test/"
train_aid_fea=${dataset_dir}"/train_uid_aid_bin.csv"
fea_dict=${dataset_dir}"/dic.pkl"
sub_file=${CURDIR}"/submission_nffm.csv"

printf '\nStep1: PreProcess UserFeature...\n'
if [ ! -f $user_feature ]; then 
    cd src
    python3 load_vowpal.py 
    printf 'Save to input/userFeature.csv\n'
    cd ..
else
    printf 'UserFeature exists, Skip this step!\n'
fi

printf '\nStep2: Make Features...\n'
if [ ! -f $train_aid_fea ]; then
    cd src
    python3 make_feature.py
    cd ..
else
    printf 'Feature exists, Skip this step!\n'
fi

printf '\nStep3: Make Dataset...\n'
if [ ! -f $fea_dict ]; then
    cd src
    python3 make_dataset.py
    cd ..
else
    printf 'Feature exists, Skip this step!\n'
fi

printf '\nStep4: Train Model and Predic Result...\n'
if [ ! -f $sub_file ]; then
    cd src
    python3 train.py
    cd ..
else
    printf 'Train Done, Skip this step!\n'
fi