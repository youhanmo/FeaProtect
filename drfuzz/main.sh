#!/bin/zsh

time_interval=180
dataset=$1

choose_col_type_list=(
'all_old'
)

if [ "$dataset" = "mobile_price"  ]; then
   dataset_col_list=(
   'px_height-sc_h-talk_time-touch_screen'
   'sc_h-talk_time-touch_screen-battery_power-four_g-three_g'
   'pc-px_height'
   )
   output_name='price_range'
   params_set=('Dnn' 'mobilePrice' 'change' 'drfuzz')
   time=360
elif [ "$dataset" = "fetal_health" ]; then
   dataset_col_list=(
   'abnormal_short_term_variability-fetal_movement-severe_decelerations-uterine_contractions'
   'uterine_contractions-mean_value_of_long_term_variability'
    'mean_value_of_short_term_variability-abnormal_short_term_variability-histogram_median-mean_value_of_long_term_variability-histogram_variance-histogram_min'
   )
   output_name='fetal_health'
   params_set=('Dnn' 'fetalHealth' 'change' 'drfuzz')
   time=360
elif [ "$dataset" = "diabetes" ]; then
   dataset_col_list=(
    'Insulin-BMI-Age'
   )
   output_name='Outcome'
   params_set=('Dnn' 'diabetes' 'change' 'drfuzz')
   time=360
elif [ "$dataset" = "customerchurn" ]; then
   dataset_col_list=(
       'Dependents-SeniorCitizen-Contract-tenure-TotalCharges-DeviceProtection'
       'Contract-tenure'
       'SeniorCitizen-StreamingTV-MultipleLines-DeviceProtection'
   )
   output_name='Churn'
   params_set=('Dnn' 'customerchurn' 'change' 'drfuzz')
   time=360

elif [ "$dataset" = "musicgenres" ]; then
   dataset_col_list=(
       'chroma_stft-mfcc14'
       'mfcc13-mfcc7-mfcc19-mfcc9-chroma_stft-mfcc12'
       'mfcc17-rmse-mfcc9-chroma_stft-mfcc19'
   )
   output_name='label'
   params_set=('Dnn' 'musicgenres' 'change' 'drfuzz')
   time=360

elif [ "$dataset" = "bean" ]; then
   dataset_col_list=(
       'ShapeFactor1-ShapeFactor3'
       'ShapeFactor3-Perimeter-MajorAxisLength-MinorAxisLength-AspectRation'
       'Solidity-roundness-ShapeFactor2-ShapeFactor4'
   )
   output_name='Class'
   params_set=('Dnn' 'bean' 'change' 'drfuzz')
   time=360

elif [ "$dataset" = "hand_gesture" ]; then
   dataset_col_list=(
       'AL-D-BI-AU-BH-O'
       'BI-W-BE-AS-BC-R-BH-AN-BK-AB-V-AF-AC-B-BB-AE-AJ-AG-X'
       'BJ-I-AL-V-AT-BD-AG-BK-R-AZ-L-F-Q-O-W-AV-A-AE-C-D-E-BE-M-AU-BA-N-X-BL-B-U'
   )
   output_name='BM'
   params_set=('Dnn' 'handgesture' 'change' 'drfuzz')
   time=360

elif [ "$dataset" = "patient" ]; then
   dataset_col_list=(
       'LEUCOCYTE'
       'MCHC-LEUCOCYTE'
       'MCV-MCHC-ERYTHROCYTE'
   )
   output_name='SOURCE'
   params_set=('Dnn' 'patient' 'change' 'drfuzz')
   time=360

elif [ "$dataset" = "climate" ]; then
    dataset_col_list=(
        'latitude-longitude-region_id-province_id'
    )
    output_name='ddd_car'
    params_set=('Dnn' 'climate' 'change' 'drfuzz')
    time=360

else
   dataset_col_list=()
fi


for dataset_col in "${dataset_col_list[@]}"
do
   for choose_col_type in "${choose_col_type_list[@]}"
   do
       nohup python main.py --dataset $dataset --model $dataset --params_set "${params_set[@]}" --output_name $output_name --dataset_col $dataset_col --model2_type ori --terminate_type time --choose_col_type $choose_col_type --time $time --time_interval $time_interval >/dev/null 2>&1 &
   done
done

