import numpy as np

def get_low_high(data, dataset_name):
    if dataset_name == 'mobile_price':
        # 设置每个特征的取值范围
        low = {
            'battery_power': 501,
            'blue': 0,
            'clock_speed': 0.5,
            'dual_sim': 0,
            'fc': 1,
            'four_g': 0,
            'int_memory': 2,
            'm_dep': 0.1,
            'mobile_wt': 80,
            'n_cores': 1,
            'pc': 2,
            'px_height': 0,
            'px_width': 0,
            'ram': 256,
            'sc_h': 5,
            'sc_w': 5,
            'talk_time': 2,
            'three_g': 0,
            'touch_screen': 0,
            'wifi': 0
        }
        high = {
            'battery_power': 1998,
            'blue': 1,
            'clock_speed': 3.0,
            'dual_sim': 1,
            'fc': 19,
            'four_g': 1,
            'int_memory': 64,
            'm_dep': 0.9,
            'mobile_wt': 200,
            'n_cores': 8,
            'pc': 20,
            'px_height': 1960,
            'px_width': 1998,
            'ram': 4000,
            'sc_h': 19,
            'sc_w': 19,
            'talk_time': 20,
            'three_g': 1,
            'touch_screen': 1,
            'wifi': 1
        }
        # 设置每个特征是否是离散型变量
        is_enum = {
            'battery_power': False,
            'blue': True,
            'clock_speed': False,
            'dual_sim': True,
            'fc': False,
            'four_g': True,
            'int_memory': False,
            'm_dep': False,
            'mobile_wt': False,
            'n_cores': False,
            'pc': False,
            'px_height': False,
            'px_width': False,
            'ram': False,
            'sc_h': False,
            'sc_w': False,
            'talk_time': False,
            'three_g': True,
            'touch_screen': True,
            'wifi': True
        }
        enum_value = dict(map(lambda kv: (kv[0], np.sort(data[kv[0]].unique())), filter(lambda kv : kv[1] is True, is_enum.items())))
        # 设置每个特征是否为整数值
        discrete = {
            'battery_power': True,
            'blue': True,
            'clock_speed': False,
            'dual_sim': True,
            'fc': True,
            'four_g': True,
            'int_memory': True,
            'm_dep': False,
            'mobile_wt': True,
            'n_cores': True,
            'pc': True,
            'px_height': True,
            'px_width': True,
            'ram': True,
            'sc_h': True,
            'sc_w': True,
            'talk_time': True,
            'three_g': True,
            'touch_screen': True,
            'wifi': True}
    elif dataset_name == 'fetal_health':
        low = {
            'baseline_value': 106,
            'accelerations': 0,
            'fetal_movement': 0,
            'uterine_contractions': 0,
            'light_decelerations': 0,
            'severe_decelerations': 0,
            'prolongued_decelerations': 0,
            'abnormal_short_term_variability': 12,
            'mean_value_of_short_term_variability': 0.2,
            'percentage_of_time_with_abnormal_long_term_variability': 0,
            'mean_value_of_long_term_variability': 0,
            'histogram_width': 3,
            'histogram_min': 50,
            'histogram_max': 122,
            'histogram_number_of_peaks': 0,
            'histogram_number_of_zeroes': 0,
            'histogram_mode': 60,
            'histogram_mean': 73,
            'histogram_median': 77,
            'histogram_variance': 0,
            'histogram_tendency': -1
        }
        high = {
            'baseline_value': 160,
            'accelerations': 0.02,
            'fetal_movement': 0.48,
            'uterine_contractions': 0.01,
            'light_decelerations': 0.01,
            'severe_decelerations': 0.0225,
            'prolongued_decelerations': 0.01,
            'abnormal_short_term_variability': 87,
            'mean_value_of_short_term_variability': 7,
            'percentage_of_time_with_abnormal_long_term_variability': 91,
            'mean_value_of_long_term_variability': 50.7,
            'histogram_width': 180,
            'histogram_min': 159,
            'histogram_max': 238,
            'histogram_number_of_peaks': 18,
            'histogram_number_of_zeroes': 10,
            'histogram_mode': 187,
            'histogram_mean': 182,
            'histogram_median': 186,
            'histogram_variance': 269,
            'histogram_tendency': 1
        }
        # 设置每个特征是否是离散型变量
        is_enum = {
            'baseline_value': False,
            'accelerations': False,
            'fetal_movement': False,
            'uterine_contractions': False,
            'light_decelerations': False,
            'severe_decelerations': False,
            'prolongued_decelerations': False,
            'abnormal_short_term_variability': False,
            'mean_value_of_short_term_variability': False,
            'percentage_of_time_with_abnormal_long_term_variability': False,
            'mean_value_of_long_term_variability': False,
            'histogram_width': False,
            'histogram_min': False,
            'histogram_max': False,
            'histogram_number_of_peaks': False,
            'histogram_number_of_zeroes': False,
            'histogram_mode': False,
            'histogram_mean': False,
            'histogram_median': False,
            'histogram_variance': False,
            'histogram_tendency': False
        }
        enum_value = dict(
            map(lambda kv: (kv[0], np.sort(data[kv[0]].unique())), filter(lambda kv: kv[1] is True, is_enum.items())))

        # 设置每个特征是否为整数值
        discrete = {
            'baseline_value': True,
            'accelerations': False,
            'fetal_movement': False,
            'uterine_contractions': False,
            'light_decelerations': False,
            'severe_decelerations': False,
            'prolongued_decelerations': False,
            'abnormal_short_term_variability': True,
            'mean_value_of_short_term_variability': False,
            'percentage_of_time_with_abnormal_long_term_variability': True,
            'mean_value_of_long_term_variability': False,
            'histogram_width': True,
            'histogram_min': True,
            'histogram_max': True,
            'histogram_number_of_peaks': True,
            'histogram_number_of_zeroes': True,
            'histogram_mode': True,
            'histogram_mean': True,
            'histogram_median': True,
            'histogram_variance': True,
            'histogram_tendency': True
        }
    elif dataset_name == "diabetes":
        low = {
            "Pregnancies": 0,
            "Glucose": 0,
            "BloodPressure": 0,
            "SkinThickness": 0,
            "Insulin": 0,
            "BMI": 0,
            "DiabetesPedigreeFunction": 0,
            "Age": 21
        }
        high = {
            "Pregnancies": 17,
            "Glucose": 199,
            "BloodPressure": 122,
            "SkinThickness": 99,
            "Insulin": 846,
            "BMI": 67.1,
            "DiabetesPedigreeFunction":2.42,
            "Age": 81
        }
        # 设置每个特征是否是离散型变量
        is_enum = {
            "Pregnancies": True,
            "Glucose": False,
            "BloodPressure": False,
            "SkinThickness": False,
            "Insulin": False,
            "BMI": False,
            "DiabetesPedigreeFunction": False,
            "Age": False
        }
        enum_value = dict(
            map(lambda kv: (kv[0], np.sort(data[kv[0]].unique())), filter(lambda kv: kv[1] is True, is_enum.items())))
        discrete = {
            "Pregnancies": True,
            "Glucose": False,
            "BloodPressure": False,
            "SkinThickness": False,
            "Insulin": False,
            "BMI": False,
            "DiabetesPedigreeFunction": False,
            "Age": True
        }
    elif dataset_name == 'customerchurn':
        low = {"gender": 0, "SeniorCitizen": 0, "Partner": 0, "Dependents": 0, "tenure": 0, "PhoneService": 0, "MultipleLines": 0, "InternetService": 0, "OnlineSecurity": 0, "OnlineBackup": 0, "DeviceProtection": 0, "TechSupport": 0, "StreamingTV": 0, "StreamingMovies": 0, "Contract": 0, "PaperlessBilling": 0, "PaymentMethod": 0, "MonthlyCharges": 18.25, "TotalCharges": 0}
        high = {"gender": 1, "SeniorCitizen": 1, "Partner": 1, "Dependents": 1, "tenure": 72, "PhoneService": 1, "MultipleLines": 2, "InternetService": 2, "OnlineSecurity": 2, "OnlineBackup": 2, "DeviceProtection": 2, "TechSupport": 2, "StreamingTV": 2, "StreamingMovies": 2, "Contract": 2, "PaperlessBilling": 1, "PaymentMethod": 3, "MonthlyCharges": 118.75, "TotalCharges": 6530}
        is_enum = {"gender": True, "SeniorCitizen": True, "Partner": True, "Dependents": True, 
                   "tenure": False, "PhoneService": True, "MultipleLines": True, "InternetService": True, 
                   "OnlineSecurity": True, "OnlineBackup": True, "DeviceProtection": True, "TechSupport": True, 
                   "StreamingTV": True, "StreamingMovies": True, "Contract": True, "PaperlessBilling": True, 
                   "PaymentMethod": True, "MonthlyCharges": False, "TotalCharges": False}
        enum_value = dict(
            map(lambda kv: (kv[0], np.sort(data[kv[0]].unique())), filter(lambda kv: kv[1] is True, is_enum.items())))
        discrete = {"gender": True, "SeniorCitizen": True, "Partner": True, "Dependents": True, 
                   "tenure": True, "PhoneService": True, "MultipleLines": True, "InternetService": True, 
                   "OnlineSecurity": True, "OnlineBackup": True, "DeviceProtection": True, "TechSupport": True, 
                   "StreamingTV": True, "StreamingMovies": True, "Contract": True, "PaperlessBilling": True, 
                   "PaymentMethod": True, "MonthlyCharges": False, "TotalCharges": True}
        
    elif dataset_name == "bean":
        low = {"Area": 21348, "Perimeter": 530.8249999999999, "MajorAxisLength": 187.1686348226686, "MinorAxisLength": 131.4330586298897, 
               "AspectRation": 1.0607980199097815, "Eccentricity": 0.333679657773177, 
               "ConvexArea": 21590, "EquivDiameter": 164.8669700122079, "Extent": 0.5666692537545113, 
               "Solidity": 0.9490232053874862, "roundness": 0.5567658261246129, "Compactness": 0.6487619599295084, 
               "ShapeFactor1": 0.0027780126683855, "ShapeFactor2": 0.0005899546965585, "ShapeFactor3": 0.4208920806515769, 
               "ShapeFactor4": 0.9499903110044128}
        high = {"Area": 254616, "Perimeter": 1985.37, "MajorAxisLength": 738.8601534818813, 
                "MinorAxisLength": 460.1984968278401, "AspectRation": 2.3640166092307124, 
                "Eccentricity": 0.9061255398367508, "ConvexArea": 263261, "EquivDiameter": 569.3743583287609, 
                "Extent": 0.8528414260898337, "Solidity": 0.9938195569962184, "roundness": 0.9866847311746488, 
                "Compactness": 0.970515523241471, "ShapeFactor1": 0.0097200558804008, "ShapeFactor2": 0.0036649719644516,
                "ShapeFactor3": 0.9419003808526664, "ShapeFactor4": 0.9997325300471388}
        is_enum = {"Area": False, "Perimeter": False, "MajorAxisLength": False, 
                   "MinorAxisLength": False, "AspectRation": False, "Eccentricity": False, "ConvexArea": False, 
                   "EquivDiameter": False, "Extent": False, "Solidity": False, "roundness": False, "Compactness": False, "ShapeFactor1": False, 
                   "ShapeFactor2": False, "ShapeFactor3": False, "ShapeFactor4": False}
        enum_value = dict(
            map(lambda kv: (kv[0], np.sort(data[kv[0]].unique())), filter(lambda kv: kv[1] is True, is_enum.items())))
        discrete = {"Area": False, "Perimeter": False, "MajorAxisLength": False, 
                   "MinorAxisLength": False, "AspectRation": False, "Eccentricity": False, "ConvexArea": False, 
                   "EquivDiameter": False, "Extent": False, "Solidity": False, "roundness": False, "Compactness": False, "ShapeFactor1": False, 
                   "ShapeFactor2": False, "ShapeFactor3": False, "ShapeFactor4": False}
        
    elif dataset_name == 'hand_gesture':
        low = {"A": -116.0, "B": -104.0, "C": -33.0, "D": -75.0, "E": -121.0, "F": -122.0, "G": -128.0, "H": -128.0, "I": -110.0, "J": -128.0, "K": -36.0, "L": -84.0, "M": -102.0, "N": -128.0, "O": -128.0, "P": -128.0, "Q": -119.0, "R": -109.0, "S": -38.0, "T": -99.0, "U": -103.0, "V": -120.0, "W": -128.0, "X": -118.0, "Y": -108.0, "Z": -115.0, "AA": -28.0, "AB": -95.0, "AC": -108.0, "AD": -128.0, "AE": -128.0, "AF": -128.0, "AG": -104.0, "AH": -112.0, "AI": -56.0, "AJ": -85.0, "AK": -95.0, "AL": -128.0, "AM": -128.0, "AN": -128.0, "AO": -121.0, "AP": -120.0, "AQ": -39.0, "AR": -79.0, "AS": -102.0, "AT": -123.0, "AU": -128.0, "AV": -128.0, "AW": -128.0, "AX": -120.0, "AY": -54.0, "AZ": -86.0, "BA": -117.0, "BB": -128.0, "BC": -128.0, "BD": -128.0, "BE": -116.0, "BF": -128.0, "BG": -46.0, "BH": -74.0, "BI": -103.0, "BJ": -128.0, "BK": -128.0, "BL": -124.0}
        high = {"A": 111.0, "B": 90.0, "C": 34.0, "D": 55.0, "E": 92.0, "F": 127.0, "G": 127.0, "H": 126.0, "I": 127.0, "J": 106.0, "K": 42.0, "L": 61.0, "M": 83.0, "N": 127.0, "O": 127.0, "P": 127.0, "Q": 127.0, "R": 118.0, "S": 34.0, "T": 47.0, "U": 90.0, "V": 127.0, "W": 127.0, "X": 108.0, "Y": 120.0, "Z": 123.0, "AA": 38.0, "AB": 44.0, "AC": 114.0, "AD": 127.0, "AE": 127.0, "AF": 127.0, "AG": 127.0, "AH": 102.0, "AI": 44.0, "AJ": 56.0, "AK": 90.0, "AL": 127.0, "AM": 127.0, "AN": 127.0, "AO": 118.0, "AP": 106.0, "AQ": 65.0, "AR": 54.0, "AS": 115.0, "AT": 127.0, "AU": 127.0, "AV": 127.0, "AW": 112.0, "AX": 111.0, "AY": 57.0, "AZ": 76.0, "BA": 92.0, "BB": 127.0, "BC": 127.0, "BD": 114.0, "BE": 127.0, "BF": 105.0, "BG": 29.0, "BH": 51.0, "BI": 110.0, "BJ": 127.0, "BK": 127.0, "BL": 127.0}
        is_enum = {"A": False, "B": False, "C": False, "D": False, "E": False, "F": False, "G": False, "H": False,
                    "I": False, "J": False, "K": False, "L": False, "M": False, "N": False, "O": False, "P": False, "Q": False, 
                    "R": False, "S": False, "T": False, "U": False, "V": False, "W": False, "X": False, "Y": False, "Z": False, 
                    "AA": False, "AB": False, "AC": False, "AD": False, "AE": False, "AF": False, "AG": False, "AH": False, "AI": False, 
                    "AJ": False, "AK": False, "AL": False, "AM": False, "AN": False, "AO": False, "AP": False, "AQ": False, "AR": False, "AS": False, 
                    "AT": False, "AU": False, "AV": False, "AW": False, "AX": False, "AY": False, "AZ": False, "BA": False, "BB": False, "BC": False, "BD": False, 
                    "BE": False, "BF": False, "BG": False, "BH": False, "BI": False, "BJ": False, "BK": False, "BL": False}
        enum_value = dict(
            map(lambda kv: (kv[0], np.sort(data[kv[0]].unique())), filter(lambda kv: kv[1] is True, is_enum.items())))
        discrete =  {"A": False, "B": False, "C": False, "D": False, "E": False, "F": False, "G": False, "H": False,
                    "I": False, "J": False, "K": False, "L": False, "M": False, "N": False, "O": False, "P": False, "Q": False, 
                    "R": False, "S": False, "T": False, "U": False, "V": False, "W": False, "X": False, "Y": False, "Z": False, 
                    "AA": False, "AB": False, "AC": False, "AD": False, "AE": False, "AF": False, "AG": False, "AH": False, "AI": False, 
                    "AJ": False, "AK": False, "AL": False, "AM": False, "AN": False, "AO": False, "AP": False, "AQ": False, "AR": False, "AS": False, 
                    "AT": False, "AU": False, "AV": False, "AW": False, "AX": False, "AY": False, "AZ": False, "BA": False, "BB": False, "BC": False, "BD": False, 
                    "BE": False, "BF": False, "BG": False, "BH": False, "BI": False, "BJ": False, "BK": False, "BL": False}
    
    elif dataset_name == "musicgenres":
        high = {"chroma_stft": 0.663572729, "rmse": 0.398011863, "spectral_centroid": 4434.439444, 
                "spectral_bandwidth": 3509.578677, "rolloff": 8676.405868, "zero_crossing_rate": 0.274829404, 
                "mfcc1": 42.03458786, "mfcc2": 193.0965118, "mfcc3": 56.6660881, "mfcc4": 80.69127655, "mfcc5": 31.46165657,
                "mfcc6": 45.17317581, "mfcc7": 21.83576775, "mfcc8": 49.01888657, "mfcc9": 19.12920952, "mfcc10": 27.21674538, 
                "mfcc11": 17.42103767, "mfcc12": 23.03757286, "mfcc13": 13.05433369, "mfcc14": 18.16166115, "mfcc15": 12.35758781, 
                "mfcc16": 13.46880245, "mfcc17": 11.48999405, "mfcc18": 15.3792572, "mfcc19": 14.68691063, "mfcc20": 15.36896706}
        low = {"chroma_stft": 0.17178233, "rmse": 0.005275614, "spectral_centroid": 569.930721, 
               "spectral_bandwidth": 897.9943189, "rolloff": 749.0621372, "zero_crossing_rate": 0.021700549, 
               "mfcc1": -552.0640259, "mfcc2": -1.527147412, "mfcc3": -89.90113831, "mfcc4": -18.76846123, "mfcc5": -38.90345001, 
               "mfcc6": -28.4245472, "mfcc7": -32.93358612, "mfcc8": -24.94753647, "mfcc9": -31.65306091, "mfcc10": -12.05118942, 
               "mfcc11": -28.05226517, "mfcc12": -15.80522537, "mfcc13": -27.54230881, "mfcc14": -12.598773, "mfcc15": -17.5454731, 
               "mfcc16": -15.69358921, "mfcc17": -17.22776604, "mfcc18": -11.97569752, "mfcc19": -18.50418663, "mfcc20": -19.93520164}
        is_enum = {"chroma_stft": False, "rmse": False, "spectral_centroid": False, "spectral_bandwidth": False, 
                   "rolloff": False, "zero_crossing_rate": False, "mfcc1": False, "mfcc2": False, "mfcc3": False, 
                   "mfcc4": False, "mfcc5": False, "mfcc6": False, "mfcc7": False, "mfcc8": False, "mfcc9": False, "mfcc10": False, 
                   "mfcc11": False, "mfcc12": False, "mfcc13": False, "mfcc14": False, "mfcc15": False, "mfcc16": False, "mfcc17": False, 
                   "mfcc18": False, "mfcc19": False, "mfcc20": False}
        enum_value = dict(
            map(lambda kv: (kv[0], np.sort(data[kv[0]].unique())), filter(lambda kv: kv[1] is True, is_enum.items())))
        discrete = {"chroma_stft": False, "rmse": False, "spectral_centroid": False, "spectral_bandwidth": False, 
                   "rolloff": False, "zero_crossing_rate": False, "mfcc1": False, "mfcc2": False, "mfcc3": False, 
                   "mfcc4": False, "mfcc5": False, "mfcc6": False, "mfcc7": False, "mfcc8": False, "mfcc9": False, "mfcc10": False, 
                   "mfcc11": False, "mfcc12": False, "mfcc13": False, "mfcc14": False, "mfcc15": False, "mfcc16": False, "mfcc17": False, 
                   "mfcc18": False, "mfcc19": False, "mfcc20": False}
    
    elif dataset_name == "patient":
        high = {"HAEMATOCRIT": 69.0, "HAEMOGLOBINS": 18.9, "ERYTHROCYTE": 7.86, "LEUCOCYTE": 76.6, "THROMBOCYTE": 1121, "MCH": 40.8, "MCHC": 38.4, "MCV": 115.6, "AGE": 99, "SEX": 1}
        low = {"HAEMATOCRIT": 13.7, "HAEMOGLOBINS": 3.8, "ERYTHROCYTE": 1.48, "LEUCOCYTE": 1.1, "THROMBOCYTE": 10, "MCH": 14.9, "MCHC": 26.0, "MCV": 54.0, "AGE": 1, "SEX": 0}
        is_enum = {"HAEMATOCRIT": False, "HAEMOGLOBINS": False, "ERYTHROCYTE": False,
                    "LEUCOCYTE": False, "THROMBOCYTE": False, "MCH": False, "MCHC": False, 
                    "MCV": False, "AGE": False, "SEX": True}
        enum_value = dict(
            map(lambda kv: (kv[0], np.sort(data[kv[0]].unique())), filter(lambda kv: kv[1] is True, is_enum.items())))
        discrete = {"HAEMATOCRIT": False, "HAEMOGLOBINS": False, "ERYTHROCYTE": False,
                    "LEUCOCYTE": False, "THROMBOCYTE": True, "MCH": False, "MCHC": False, 
                    "MCV": False, "AGE": True, "SEX": True}
    

    elif dataset_name == "climate":
        high = {"Tn": 33.0, "Tx": 38.7, "Tavg": 31.8, 
                "RH_avg": 100, "RR": 235.0, "ss": 88.0, 
                "ff_x": 46, "ddd_x": 888, "ff_avg": 54, 
                "latitude": 5.87655, "longitude": 125.52881, "region_id": 507, "province_id": 34}
        low = {"Tn": 0.0, "Tx": 19.1, "Tavg": 15.3, "RH_avg": 35, "RR": 0.0, "ss": 0.0, "ff_x": 0, "ddd_x": 0, "ff_avg": 0, "latitude": -8.75, "longitude": 95.33785, "region_id": 1, "province_id": 1}
        is_enum = {"Tn": False, "Tx": False, "Tavg": False, "RH_avg": False, "RR": False, "ss": False, "ff_x": False, "ddd_x": False, "ff_avg": False, "latitude": False, "longitude": False, "region_id": False, 
                   "province_id": False}
        
        enum_value = dict(
            map(lambda kv: (kv[0], np.sort(data[kv[0]].unique())), filter(lambda kv: kv[1] is True, is_enum.items())))
        
        discrete = {"Tn": False, "Tx": False, "Tavg": False, 
                    "RH_avg": False, "RR": False, "ss": False, 
                    "ff_x": False, "ddd_x": False, "ff_avg": False, 
                    "latitude": False, "longitude": False, "region_id": True, 
                   "province_id": True}
    
    return low, high, is_enum, enum_value, discrete