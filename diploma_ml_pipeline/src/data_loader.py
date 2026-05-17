# src/data_loader.py

import pandas as pd
import os
import urllib.request
import zipfile

# ==========================================
# ДИНАМИЧНИ ПЪТИЩА (УНИВЕРСАЛНИ)
# ==========================================
# 1. Намираме папката 'src', където се намира текущия файл
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))

# 2. Връщаме се една директория назад, към главната папка на проекта
BASE_DIR = os.path.dirname(CURRENT_DIR)

# 3. Дефинираме главната папка за сурови данни
RAW_DATA_DIR = os.path.join(BASE_DIR, "data", "raw")

# ==========================================
# 1. Зареждане на Таблични данни (Adult Census)
# ==========================================
def download_and_load_adult_data():
    """
    Автоматично изтегля Adult Census Income от UCI Repository 
    и го превръща в Pandas DataFrame.
    """
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data"
    data_dir = RAW_DATA_DIR
    file_path = os.path.join(data_dir, "adult.csv")
    
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
    
    if not os.path.exists(file_path):
        print("🌐 Изтегляне на Adult Census Income от UCI...")
        urllib.request.urlretrieve(url, file_path)
        print("✅ Изтеглянето завърши.")
    
    # Имена на колоните (липсват в суровия файл)
    columns = [
        "age", "workclass", "fnlwgt", "education", "education-num",
        "marital-status", "occupation", "relationship", "race", "sex",
        "capital-gain", "capital-loss", "hours-per-week", "native-country", "income"
    ]
    
    # ' ?' се среща често в този dataset като индикатор за липсващи данни
    df = pd.read_csv(file_path, names=columns, sep=',', skipinitialspace=True, na_values='?')
    
    # Превръщаме таргета в бинарен (0 за <=50K, 1 за >50K)
    df['income'] = df['income'].apply(lambda x: 1 if x == '>50K' else 0)
    
    return df

def get_feature_lists():
    """Връща списъци с числови и категорийни колони за Adult dataset."""
    numeric = ["age", "education-num", "capital-gain", "capital-loss", "hours-per-week"]
    categorical = ["workclass", "education", "marital-status", "occupation", "relationship", "race", "sex", "native-country"]
    return numeric, categorical

# ==========================================
# 2. Зареждане на Текстови данни (SMS Spam)
# ==========================================
def download_and_load_sms_data():
    """
    Автоматично изтегля SMS Spam Collection от UCI Repository,
    разархивира го и го подготвя като Pandas DataFrame.
    """
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00228/smsspamcollection.zip"
    data_dir = RAW_DATA_DIR
    zip_path = os.path.join(data_dir, "smsspamcollection.zip")
    extract_dir = os.path.join(data_dir, "sms_spam")
    
    os.makedirs(data_dir, exist_ok=True)
    
    # Изтегляне и разархивиране, ако вече не съществува
    if not os.path.exists(zip_path):
        print("🌐 Изтегляне на SMS Spam Collection от UCI...")
        urllib.request.urlretrieve(url, zip_path)
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(extract_dir)
        print("✅ Изтеглянето и разархивирането завършиха.")
    
    # Файлът е отделен с табулация (TSV)
    file_path = os.path.join(extract_dir, "SMSSpamCollection")
    
    # Зареждане на данните
    df = pd.read_csv(file_path, sep='\t', names=["label", "text"])
    
    # Преобразуване на етикетите в бинарен формат (0: ham/нормален, 1: spam)
    df['target'] = df['label'].map({'ham': 0, 'spam': 1})
    
    return df[['text', 'target']]