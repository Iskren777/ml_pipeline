# src/nlp_analyzer.py

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

def plot_tfidf_feature_importance(best_pipeline, top_n=15, output_dir="C:\\Users\\kottk\\Desktop\\Много важно\\diploma_ml_pipeline\\results\\figures"):
    """
    Извлича най-важните думи от TfidfVectorizer и LogisticRegression
    и генерира графика (Feature Importance).
    """
    print("\n🧠 Анализ на най-важните думи (TF-IDF Feature Importance)...")
    os.makedirs(output_dir, exist_ok=True)

    # 1. Извличане на компонентите от вече тренирания Pipeline
    vectorizer = best_pipeline.named_steps['vectorizer']
    model = best_pipeline.named_steps['model']

    # Проверка дали моделът е линеен (има коефициенти)
    if not hasattr(model, 'coef_'):
        print("⚠️ Този модел няма 'coef_' атрибут. Изберете LogisticRegression за този анализ.")
        return

    # 2. Вземане на думите (речника) и техните тежести
    feature_names = vectorizer.get_feature_names_out()
    coefficients = model.coef_[0] # Вземаме коефициентите (за бинарна класификация е първият масив)

    # Създаване на DataFrame
    importance_df = pd.DataFrame({
        'Word': feature_names,
        'Coefficient': coefficients
    })

    # 3. Намиране на най-силните думи за Спам (положителни) и Нормални (отрицателни)
    top_spam_words = importance_df.sort_values(by='Coefficient', ascending=False).head(top_n)
    top_ham_words = importance_df.sort_values(by='Coefficient', ascending=True).head(top_n)

    # Обединяване на данните за графиката
    top_words = pd.concat([top_spam_words, top_ham_words]).sort_values(by='Coefficient', ascending=False)

    # Добавяне на колона за цвят/категория
    top_words['Категория'] = top_words['Coefficient'].apply(lambda x: 'Спам индикатор' if x > 0 else 'Нормално съобщение')

    # 4. Генериране на графиката
    plt.figure(figsize=(12, 8))
    sns.set_theme(style="whitegrid")
    
    # Използваме червено за Спам и зелено за нормални съобщения
    custom_palette = {'Спам индикатор': '#d62728', 'Нормално съобщение': '#2ca02c'}
    
    sns.barplot(
        data=top_words, 
        x='Coefficient', 
        y='Word', 
        hue='Категория',
        palette=custom_palette
    )

    plt.title(f'Топ {top_n} думи, определящи Спам срещу Нормални съобщения', fontsize=16, pad=15)
    plt.xlabel('Тежест на думата (Коефициент от Logistic Regression)', fontsize=12)
    plt.ylabel('Дума (Feature)', fontsize=12)
    plt.tight_layout()

    # Запазване
    output_file = os.path.join(output_dir, 'tfidf_feature_importance.png')
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✅ Графиката с думите е запазена в: {output_file}")
    
    return top_words