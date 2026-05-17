# src/evaluator.py

import os
import pandas as pd
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report, accuracy_score, f1_score

# ==========================================
# ДИНАМИЧНИ ПЪТИЩА
# ==========================================
# 1. Намираме папката 'src'
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))

# 2. Връщаме се една директория назад към главната папка
BASE_DIR = os.path.dirname(CURRENT_DIR)

# 3. Дефинираме главната папка за резултатите (метрики)
DEFAULT_METRICS_DIR = os.path.join(BASE_DIR, "results", "metrics")


def clean_cv_results(cv_results: dict) -> pd.DataFrame:
    """
    Специфично почистване на резултатите за ML (Таблични данни).
    Генерира красив контраст: извлича Базовия модел (Преди) и само ТОП 10 най-добри (След).
    """
    df_results = pd.DataFrame(cv_results)
    df_results['Модел'] = df_results['param_model'].apply(
        lambda x: x.__class__.__name__ if pd.notnull(x) else 'Unknown'
    )
    
    # 1. Интелигентно етикетиране на трансформациите (ПРЕДИ и СЛЕД)
    def format_scaler(scaler_obj):
        if scaler_obj == 'passthrough' or scaler_obj is None or pd.isna(scaler_obj):
            return 'ПРЕДИ (Без мащабиране)'
        else:
            return f"СЛЕД ({scaler_obj.__class__.__name__})"
            
    if 'param_preprocessor__num__scaler' in df_results.columns:
        df_results['Етап / Трансформация'] = df_results['param_preprocessor__num__scaler'].apply(format_scaler)
    else:
        df_results['Етап / Трансформация'] = 'Стандартен'

    df_results['Точност (Accuracy)'] = df_results['mean_test_accuracy'].round(4)
    df_results['F1-Score'] = df_results['mean_test_f1_macro'].round(4)
    
    # 2. Кристално четими допълнителни настройки
    def get_clean_params(params_dict):
        model_params = []
        for k, v in params_dict.items():
            if k in ['model', 'preprocessor__num__scaler']:
                continue
            
            val_str = "Без" if v is None else str(v)
            if 'model__' in k:
                key = k.replace('model__', '')
                model_params.append(f"{key}={val_str}")
                
        if model_params:
            return "Модел: [" + ", ".join(model_params) + "]"
        return "По подразбиране"

    df_results['Доп. настройки (Модел)'] = df_results['params'].apply(get_clean_params)
    
    cols = ['Модел', 'Етап / Трансформация', 'Точност (Accuracy)', 'F1-Score', 'Доп. настройки (Модел)']
    df_clean = df_results[cols].copy()
    
    # ========================================================
    # 3. РАЗДЕЛЯНЕ НА ПРЕДИ И СЛЕД + ИЗВЛИЧАНЕ НА ТОП 10
    # ========================================================
    
    # Отделяме резултатите ПРЕДИ мащабиране
    baseline_df = df_clean[df_clean['Етап / Трансформация'] == 'ПРЕДИ (Без мащабиране)']
    best_baseline = baseline_df.sort_values('Точност (Accuracy)', ascending=False).groupby('Модел').head(1)
    
    # Отделяме резултатите СЛЕД мащабиране
    after_df = df_clean[df_clean['Етап / Трансформация'] != 'ПРЕДИ (Без мащабиране)']
    top_10_after = after_df.sort_values(['Точност (Accuracy)', 'F1-Score'], ascending=[False, False]).head(10)
    
    # Обединяваме и сортираме наново, за да се подреди правилният ранк
    final_df = pd.concat([best_baseline, top_10_after])
    final_df = final_df.sort_values(['Точност (Accuracy)', 'F1-Score'], ascending=[False, False]).reset_index(drop=True)
    
    # Добавяме ранкинг от 1 надолу
    final_df.index = final_df.index + 1
    final_df.index.name = 'Ранк'
    final_df = final_df.reset_index()
    
    return final_df


def clean_nlp_cv_results(cv_results: dict) -> pd.DataFrame:
    """
    Специфично почистване на резултатите за NLP. 
    Генерира красив контраст, реално сортиран ранк и четими хиперпараметри.
    """
    df_results = pd.DataFrame(cv_results)
    df_results['Модел'] = df_results['param_model'].apply(lambda x: x.__class__.__name__)
    
    # 1. Интелигентно етикетиране на текстовото почистване
    def format_nlp_cleaning(row):
        p = row.get('params', {})
        active_methods = []
        
        if p.get('cleaner__replace_urls', False): active_methods.append('URL->Токен')
        if p.get('cleaner__replace_emails', False): active_methods.append('Email->Токен')
        if p.get('cleaner__replace_numbers', False): active_methods.append('Числа->Токен')
        if p.get('cleaner__remove_punctuation', False): active_methods.append('Без пунктуация')
        
        if not active_methods:
            # Проверяваме дали това е Истинският Базов Модел (без TF-IDF оптимизации)
            is_default_tfidf = (p.get('vectorizer__min_df', 1) == 1 and 
                                p.get('vectorizer__sublinear_tf', False) == False)
            if is_default_tfidf:
                return 'АБСОЛЮТЕН БАЗОВ МОДЕЛ (Default)'
            else:
                return 'ПРЕДИ (Суров текст + Оптимизиран TF-IDF)'
        else:
            return 'СЛЕД (' + ', '.join(active_methods) + ')'

    df_results['Етап / Трансформация'] = df_results.apply(format_nlp_cleaning, axis=1)
    df_results['Точност (Accuracy)'] = df_results['mean_test_accuracy'].round(4)
    df_results['F1-Score'] = df_results['mean_test_f1_macro'].round(4)
    
    # ========================================================
    # ПОДОБРЕНИЕ 1: Кристално четими допълнителни настройки
    # ========================================================
    def get_clean_params(params_dict):
        model_params = []
        tfidf_params = []
        
        for k, v in params_dict.items():
            if k == 'model' or 'cleaner__' in k:
                continue
            
            # Правим стойностите по-четими (пр. None става 'Без')
            val_str = "Без" if v is None else str(v)
            
            if 'model__' in k:
                key = k.replace('model__', '')
                model_params.append(f"{key}={val_str}")
            elif 'vectorizer__' in k:
                key = k.replace('vectorizer__', '')
                tfidf_params.append(f"{key}={val_str}")
        
        # Групираме ги красиво
        parts = []
        if model_params:
            parts.append("Модел: [" + ", ".join(model_params) + "]")
        if tfidf_params:
            parts.append("TF-IDF: [" + ", ".join(tfidf_params) + "]")
            
        return " | ".join(parts)

    df_results['Доп. настройки (TF-IDF/Модел)'] = df_results['params'].apply(get_clean_params)
    
    cols = ['Модел', 'Етап / Трансформация', 'Точност (Accuracy)', 'F1-Score', 'Доп. настройки (TF-IDF/Модел)']
    df_clean = df_results[cols].copy()
    
    # ========================================================
    # 2. РАЗДЕЛЯНЕ НА ПРЕДИ И СЛЕД + ИЗВЛИЧАНЕ НА ТОП 10
    # ========================================================
    
    # Отделяме АБСОЛЮТНИЯ БАЗОВ МОДЕЛ
    baseline_df = df_clean[df_clean['Етап / Трансформация'] == 'АБСОЛЮТЕН БАЗОВ МОДЕЛ (Default)']
    best_baseline = baseline_df.sort_values('Точност (Accuracy)', ascending=False).groupby('Модел').head(1)
    
    # Отделяме ТОП 10 резултатите СЛЕД интелигентна обработка
    after_df = df_clean[df_clean['Етап / Трансформация'].str.startswith('СЛЕД')]
    top_10_after = after_df.sort_values(['Точност (Accuracy)', 'F1-Score'], ascending=[False, False]).head(10)
    
    # Обединяваме и сортираме
    final_df = pd.concat([best_baseline, top_10_after])
    final_df = final_df.sort_values(['Точност (Accuracy)', 'F1-Score'], ascending=[False, False]).reset_index(drop=True)
    
    # Добавяме ранкинг от 1 надолу
    final_df.index = final_df.index + 1
    final_df.index.name = 'Ранк'
    final_df = final_df.reset_index()
    
    return final_df


def run_evaluation(pipeline, param_grid, X_train, y_train, X_test, y_test, output_csv=None, is_nlp=False):
    """
    Стартира крос-валидация (GridSearchCV), оценява най-добрия модел 
    върху тестовия сет и експортира резултатите в красив CSV формат.
    
    Параметърът is_nlp определя коя функция за почистване на CSV да се ползва.
    """

    if output_csv is None:
        filename = "nlp_experiment_results.csv" if is_nlp else "ml_experiment_results.csv"
        output_csv = os.path.join(DEFAULT_METRICS_DIR, filename)

    print("🚀 Стартиране на експериментите с GridSearchCV...")
    
    # Използваме няколко метрики едновременно: Accuracy и F1-Macro
    scoring_metrics = ['accuracy', 'f1_macro']
    
    grid_search = GridSearchCV(
        estimator=pipeline,
        param_grid=param_grid,
        cv=5, # 5-Fold Cross Validation
        scoring=scoring_metrics,
        refit='accuracy', # Накрая претренира модела с най-добрата точност
        n_jobs=-1, # Използва всички ядра на процесора за бързина
        verbose=1
    )
    
    # Трениране на всички комбинации
    grid_search.fit(X_train, y_train)
    
    print("\n✅ Тренировката приключи!")
    
    # Динамично извличане на имената на модела и трансформатора
    best_est = grid_search.best_estimator_
    model_name = best_est.named_steps['model'].__class__.__name__
    print(f"🏆 Най-добър модел (според Accuracy): {model_name}")
    
    # Оценка върху тестовите данни (Hold-out validation)
    print("\n📊 Оценка върху тестовите данни (Test Set):")
    y_pred = grid_search.predict(X_test)
    test_accuracy = accuracy_score(y_test, y_pred)
    test_f1 = f1_score(y_test, y_pred, average='macro')
    
    print(f"Test Accuracy: {test_accuracy:.4f}")
    print(f"Test F1-Macro: {test_f1:.4f}")
    print("\nДетайлен репорт на класификацията:")
    print(classification_report(y_test, y_pred))
    
    # Извличане и почистване на резултатите (избор спрямо типа проект)
    if is_nlp:
        df_clean_results = clean_nlp_cv_results(grid_search.cv_results_)
    else:
        df_clean_results = clean_cv_results(grid_search.cv_results_)
    
    # Създаване на директория и запазване в CSV (с utf-8-sig за правилна кирилица в Excel)
    import os
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    df_clean_results.to_csv(output_csv, index=False, encoding='utf-8-sig')
    
    print(f"\n📁 Пълните резултати са форматирани и запазени в: {output_csv}")
    print("Можете да отворите този CSV файл директно в Excel и кирилицата ще се чете перфектно!\n")
    
    return best_est, df_clean_results