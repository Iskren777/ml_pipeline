import shap
import pandas as pd
import matplotlib.pyplot as plt
import os

def explain_best_model(best_pipeline, X_train, output_dir="results/figures/"):
    print("🧠 Генериране на SHAP анализ за най-добрия модел...")
    os.makedirs(output_dir, exist_ok=True)
    
    preprocessor = best_pipeline.named_steps['preprocessor']
    model = best_pipeline.named_steps['model']
    
    # ТРАНСФОРМАЦИЯ: Добавяме .toarray() за справяне със sparse matrices
    X_train_transformed = preprocessor.transform(X_train)
    if hasattr(X_train_transformed, "toarray"):
        X_train_transformed = X_train_transformed.toarray()
    
    # Извличане на имената на колоните
    try:
        feature_names = preprocessor.get_feature_names_out()
        feature_names = [f.split('__')[-1] for f in feature_names]
    except Exception:
        feature_names = [f"Feature {i}" for i in range(X_train_transformed.shape[1])]

    # Създаваме DataFrame с трансформираните данни и правилните имена
    X_train_df = pd.DataFrame(X_train_transformed, columns=feature_names)

    # 4. Инициализация на SHAP Explainer според типа модел
    if type(model).__name__ == "RandomForestClassifier":
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_train_df)
        # При класификация с гора, shap_values е лист. Вземаме стойностите за клас 1
        if isinstance(shap_values, list):
            shap_values = shap_values[1]
    elif type(model).__name__ == "LogisticRegression":
        explainer = shap.LinearExplainer(model, X_train_df)
        shap_values = explainer.shap_values(X_train_df)
    else:
        print("⚠️ Този тип модел все още не се поддържа за автоматичен SHAP анализ в този скрипт.")
        return

    # 5. Генериране и запазване на SHAP Summary Plot
    plt.figure(figsize=(10, 8))
    shap.summary_plot(shap_values, X_train_df, show=False)
    
    output_file = os.path.join(output_dir, 'shap_summary_plot.png')
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✅ SHAP графиката (Feature Importance) е запазена в: {output_file}")