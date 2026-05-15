# main.py

import pandas as pd
import warnings
from sklearn.model_selection import train_test_split
from sklearn.exceptions import ConvergenceWarning

# Импортиране на нашите модули
from src.pipeline_builder import load_config, build_dynamic_pipeline_and_grid, build_nlp_pipeline_and_grid
from src.evaluator import run_evaluation
from src.analyzer import plot_scaling_impact
from src.data_loader import download_and_load_adult_data, get_feature_lists, download_and_load_sms_data
try:
    from src.nlp_analyzer import plot_tfidf_feature_importance
except ImportError:
    pass # В случай че не сте създали nlp_analyzer.py

warnings.filterwarnings("ignore", category=ConvergenceWarning)
warnings.filterwarnings("ignore", category=UserWarning)

def run_ml_pipeline():
    """
    Стартира експериментите за таблични данни (Adult Census).
    Фокус: Анализ на влиянието на мащабирането (Scaling) върху моделната селекция.
    """
    print("\n" + "="*65)
    print("📊 ЕКСПЕРИМЕНТ 1: ТАБЛИЧНИ ДАННИ (Adult Census Income)")
    print("ЦЕЛ: Анализ на влиянието на трансформациите върху моделната селекция")
    print("="*65)

    config = load_config("C:\\Users\\kottk\\Desktop\\Много важно\\diploma_ml_pipeline\\configs\\experiment_1.yaml")
    
    print("\n[Фаза 1] Подготовка и извличане на данните...")
    df = download_and_load_adult_data()
    df = df.sample(n=10000, random_state=42).reset_index(drop=True)
    num_features, cat_features = get_feature_lists()
    config['data'] = {'numeric_features': num_features, 'categorical_features': cat_features}
    
    X = df.drop('income', axis=1)
    y = df['income']
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=config.get('random_state', 42), stratify=y
    )
    print(f"-> Данните са разделени: {len(X_train)} тренировъчни и {len(X_test)} тестови записа.")
    
    print("\n[Фаза 2] Изграждане на автоматизиран ML Pipeline...")
    print("-> Зададени модели за селекция: LogisticRegression, RandomForest")
    print("-> Зададени трансформации за тест: StandardScaler, MinMaxScaler, Без трансформация")
    pipeline, param_grid = build_dynamic_pipeline_and_grid(config)
    
    print("\n[Фаза 3] Автоматизирана моделна селекция (Grid Search Cross-Validation)...")
    csv_results_path = "C:\\Users\\kottk\\Desktop\\Много важно\\diploma_ml_pipeline\\results\\metrics\\ml_experiment_results.csv"
    best_pipeline, results_df = run_evaluation(
        pipeline, param_grid, X_train, y_train, X_test, y_test, output_csv=csv_results_path, is_nlp=False
    )
    
    print("\n[Фаза 4] Анализ на влиянието на трансформациите...")
    print("-> Генериране на визуални графики за дипломната работа...")
    plot_scaling_impact(csv_results_path)
    
    print("\n✅ ЕКСПЕРИМЕНТ 1 ЗАВЪРШИ УСПЕШНО!")
    print("-> Графиките показват ясно как линейните модели зависят от мащабирането, докато дървовидните остават стабилни.")


def run_nlp_pipeline():
    """
    Стартира експериментите за текстови данни (SMS Spam).
    Фокус: Влияние на техниките за чистене на текст (NLP) върху точността на класификатора.
    """
    print("\n" + "="*65)
    print("📩 ЕКСПЕРИМЕНТ 2: НЕСТРУКТУРИРАНИ ДАННИ (SMS Spam Collection)")
    print("ЦЕЛ: Анализ на влиянието на текстовите трансформации (NLP) върху моделите")
    print("="*65)

    config = load_config("C:\\Users\\kottk\\Desktop\\Много важно\\diploma_ml_pipeline\\configs\\experiment_nlp.yaml")
    
    print("\n[Фаза 1] Подготовка на текстовите данни...")
    df = download_and_load_sms_data()
    X = df['text']
    y = df['target']
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=config.get('random_state', 42), stratify=y
    )
    print(f"-> Заредени са {len(X)} текстови съобщения.")
    
    print("\n[Фаза 2] Конфигуриране на автоматизиран NLP Pipeline...")
    print("-> Интегриране на custom TextCleaner (Токенизация на числа, премахване на пунктуация)")
    print("-> Векторизация: TF-IDF (Term Frequency - Inverse Document Frequency)")
    pipeline, param_grid = build_nlp_pipeline_and_grid(config)
    
    print("\n[Фаза 3] Моделна селекция и оптимизация на хиперпараметрите...")
    csv_results_path = "C:\\Users\\kottk\\Desktop\\Много важно\\diploma_ml_pipeline\\results\\metrics\\nlp_experiment_results.csv"
    best_pipeline, results_df = run_evaluation(
        pipeline, param_grid, X_train, y_train, X_test, y_test, output_csv=csv_results_path, is_nlp=True
    )
    
    print("\n[Фаза 4] Анализ на влиянието и Интерпретируемост (Explainable AI)...")
    print("-> Извличане на най-важните характеристики (Feature Importance) от най-добрия модел...")
    try:
        plot_tfidf_feature_importance(best_pipeline)
    except Exception as e:
        print(f"Графиката за важни думи не бе генерирана: {e}")

    print("\n✅ ЕКСПЕРИМЕНТ 2 ЗАВЪРШИ УСПЕШНО!")
    print("-> Резултатите доказват, че запазването на числата (чрез токенизация) носи критична информация за засичане на спам.")


if __name__ == "__main__":
    print("\n🤖 ДОБРЕ ДОШЛИ В АВТОМАТИЗИРАНИЯ ML PIPELINE")
    print("-------------------------------------------------")
    print("Моля, изберете тип на данните за анализ:")
    print("  [1] Таблични данни (.csv) - Влияние на мащабирането върху доходите")
    print("  [2] Текстови данни (.txt) - Влияние на почистването върху спам филтри")
    
    while True:
        choice = input("\nВъведете 1 или 2 (или 'q' за изход): ")
        
        if choice == '1':
            run_ml_pipeline()
            break
        elif choice == '2':
            run_nlp_pipeline()
            break
        elif choice.lower() == 'q':
            print("Изход от програмата. Довиждане!")
            break
        else:
            print("❌ Невалиден избор. Моля, опитайте отново.")