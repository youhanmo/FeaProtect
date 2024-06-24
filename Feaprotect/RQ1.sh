#!/bin/zsh

fix_type=$1
dataset=$2
lamb=$3


if [ "$dataset" = "mobile_price"  ]; then
    dataset_col_list=(
    'pc-px_height'
    'px_height-sc_h-talk_time-touch_screen'
    'sc_h-talk_time-touch_screen-battery_power-four_g-three_g'
    )
    num_classes=4
elif [ "$dataset" = "fetal_health" ]; then
    dataset_col_list=(
    'abnormal_short_term_variability-fetal_movement-severe_decelerations-uterine_contractions'
    'uterine_contractions-mean_value_of_long_term_variability'
    'mean_value_of_short_term_variability-abnormal_short_term_variability-histogram_median-mean_value_of_long_term_variability-histogram_variance-histogram_min'
    )
    num_classes=3
elif [ "$dataset" = "diabetes" ]; then
    dataset_col_list=(
        'BMI'
        'Glucose'
        'Glucose-Age'
        'Glucose-SkinThickness-DiabetesPedigreeFunction'
    )
    num_classes=2
elif [ "$dataset" = "customerchurn" ]; then
    dataset_col_list=(
        'Dependents-SeniorCitizen-Contract-tenure-TotalCharges-DeviceProtection'
        'Contract-tenure'
        'SeniorCitizen-StreamingTV-MultipleLines-DeviceProtection'
    )
    num_classes=2

elif [ "$dataset" = "musicgenres" ]; then
    dataset_col_list=(
        'chroma_stft-mfcc14'
        'mfcc13-mfcc7-mfcc19-mfcc9-chroma_stft-mfcc12'
        'mfcc17-rmse-mfcc9-chroma_stft-mfcc19'
    )
    num_classes=10

elif [ "$dataset" = "hand_gesture" ]; then
    dataset_col_list=(
        'AL-D-BI-AU-BH-O'
        'BI-W-BE-AS-BC-R-BH-AN-BK-AB-V-AF-AC-B-BB-AE-AJ-AG-X'
        'BJ-I-AL-V-AT-BD-AG-BK-R-AZ-L-F-Q-O-W-AV-A-AE-C-D-E-BE-M-AU-BA-N-X-BL-B-U'
    )
    num_classes=4

elif [ "$dataset" = "bean" ]; then
    dataset_col_list=(
        'ShapeFactor1-ShapeFactor3'
        'ShapeFactor3-Perimeter-MajorAxisLength-MinorAxisLength-AspectRation'
        'Solidity-roundness-ShapeFactor2-ShapeFactor4'
    )
    num_classes=7

elif [ "$dataset" = "patient" ]; then
    dataset_col_list=(
        'LEUCOCYTE'
        'MCHC-LEUCOCYTE'
        'MCV-MCHC-ERYTHROCYTE'
    )
    num_classes=2

elif [ "$dataset" = "climate" ]; then
    dataset_col_list=(
        'latitude-longitude-region_id-province_id'
    )
    num_classes=9

else
    dataset_col_list=()
fi


for dataset_col in "${dataset_col_list[@]}"
do
    log_name="${fix_type}.log"
    log_path="results/logs/${dataset}/${dataset_col}"
    if [ ! -d "$log_path" ]; then
        mkdir -p "$log_path"
    fi
    save_path="${log_path}/${log_name}"
    nohup python -u train.py --lamb $lamb --fix_type $fix_type --dataset $dataset --model $dataset --num_classes $num_classes --dataset_col $dataset_col > $save_path 2>&1 &
    
done