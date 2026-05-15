# main.py

import pandas as pd
from sklearn.model_selection import train_test_split

# Импортиране на нашите модули
from src.pipeline_builder import load_config, build_dynamic_pipeline_and_grid, build_nlp_pipeline_and_grid
from src.evaluator import run_evaluation
from src.analyzer import plot_scaling_impact
from src.data_loader import download_and_load_adult_data, get_feature_lists, download_and_load_sms_data
try:
    from src.nlp_analyzer import plot_tfidf_feature_importance
except ImportError:
    pass # В случай че не сте създали nlp_analyzer.py

def run_ml_pipeline():
    """Стартира експериментите за таблични данни (Adult Census / CSV)"""
    print("\n" + "="*50)
    print("📊 СТАРТИРАНЕ НА ML PIPELINE (Таблични данни)")
    print("="*50)

    config = load_config("C:\\Users\\kottk\\Desktop\\Много важно\\diploma_ml_pipeline\\configs\\experiment_1.yaml")
    df = download_and_load_adult_data()
    
    num_features, cat_features = get_feature_lists()
    config['data'] = {'numeric_features': num_features, 'categorical_features': cat_features}
    
    X = df.drop('income', axis=1)
    y = df['income']
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=config.get('random_state', 42), stratify=y
    )
    
    pipeline, param_grid = build_dynamic_pipeline_and_grid(config)
    
    csv_results_path = "C:\\Users\\kottk\\Desktop\\Много важно\\diploma_ml_pipeline\\results\\metrics\\ml_experiment_results.csv"
    best_pipeline, results_df = run_evaluation(
        pipeline, param_grid, X_train, y_train, X_test, y_test, output_csv=csv_results_path, is_nlp=False
    )
    
    plot_scaling_impact(csv_results_path)
    print("\n✅ Анализът на табличните данни приключи успешно!")


def run_nlp_pipeline():
    """Стартира експериментите за текстови данни (SMS Spam / TXT)"""
    print("\n" + "="*50)
    print("📩 СТАРТИРАНЕ НА NLP PIPELINE (Текстови данни)")
    print("="*50)

    config = load_config("C:\\Users\\kottk\\Desktop\\Много важно\\diploma_ml_pipeline\\configs\\experiment_nlp.yaml")
    df = download_and_load_sms_data()
    
    X = df['text']
    y = df['target']
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=config.get('random_state', 42), stratify=y
    )
    
    pipeline, param_grid = build_nlp_pipeline_and_grid(config)
    
    csv_results_path = "C:\\Users\\kottk\\Desktop\\Много важно\\diploma_ml_pipeline\\results\\metrics\\nlp_experiment_results.csv"
    best_pipeline, results_df = run_evaluation(
        pipeline, param_grid, X_train, y_train, X_test, y_test, output_csv=csv_results_path, is_nlp=True
    )
    
    # Генериране на графика за важните думи (ако модулът съществува)
    try:
        plot_tfidf_feature_importance(best_pipeline)
    except Exception as e:
        print(f"Графиката за важни думи не бе генерирана: {e}")

    print("\n✅ NLP анализът приключи успешно!")


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