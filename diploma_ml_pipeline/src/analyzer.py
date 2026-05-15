# src/analyzer.py

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os
import numpy as np

def plot_scaling_impact(csv_path: str, output_dir: str = "C:\\Users\\kottk\\Desktop\\Много важно\\diploma_ml_pipeline\\results\\figures"):
    """
    Генерира висококачествена, академична графика, показваща влиянието на мащабирането.
    Динамично мащабира Y-оста и поставя точните стойности върху колонките.
    """
    print(f"📊 Генериране на подобрена графика от {csv_path}...")
    
    # 1. Зареждане на резултатите
    df = pd.read_csv(csv_path)
    os.makedirs(output_dir, exist_ok=True)
    
    # 2. Настройка на стила (По-научен и изчистен вид)
    sns.set_theme(style="whitegrid", context="paper", font_scale=1.2)
    plt.figure(figsize=(14, 8))
    
    # 3. Изграждане на Bar Chart
    # Използваме 'Set2' или 'muted' за по-професионални цветове и слагаме черен контур (edgecolor)
    ax = sns.barplot(
        data=df, 
        x='Модел', 
        y='Точност (Accuracy)', 
        hue='Етап / Трансформация',
        palette='Set2', 
        edgecolor='black',
        linewidth=1
    )
    
    plt.title('Влияние на трансформациите върху точността на моделите', fontsize=18, fontweight='bold', pad=20)
    plt.xlabel('Машинен алгоритъм', fontsize=14, fontweight='bold', labelpad=15)
    plt.ylabel('Точност (Accuracy)', fontsize=14, fontweight='bold', labelpad=15)
    
    # ==========================================================
    # ТРИК 1: ДИНАМИЧНА Y-ОС ЗА МАКСИМАЛЕН КОНТРАСТ
    # ==========================================================
    min_acc = df['Точност (Accuracy)'].min()
    max_acc = df['Точност (Accuracy)'].max()
    
    # Настройваме графиката да показва само релевантната част (напр. от 0.70 до 0.88)
    # Това ще направи разликите очевидни!
    plt.ylim(max(0, min_acc - 0.04), min(1.0, max_acc + 0.04))
    
    # ==========================================================
    # ТРИК 2: ИЗПИСВАНЕ НА СТОЙНОСТИТЕ ВЪРХУ КОЛОНКИТЕ
    # ==========================================================
    for p in ax.patches:
        height = p.get_height()
        # Проверка дали височината е валидно число (понякога има празни места)
        if not np.isnan(height) and height > 0:
            ax.annotate(f'{height:.4f}',
                        (p.get_x() + p.get_width() / 2., height),
                        ha='center', va='bottom',
                        fontsize=11, fontweight='bold', color='black', 
                        xytext=(0, 6), textcoords='offset points', 
                        rotation=45) # Завъртаме текста леко, за да не се застъпва
    
    # 4. Оформление на легендата (Изнасяме я извън графиката, за да не скрива данните)
    plt.legend(
        title='Етап / Трансформация', 
        title_fontsize='13', 
        fontsize='11',
        bbox_to_anchor=(1.02, 1), 
        loc='upper left', 
        frameon=True, 
        shadow=True
    )
    
    plt.tight_layout()
    
    # 5. Запазване на файла с високо качество (dpi=300 е стандарт за печат на дипломни работи)
    output_file = os.path.join(output_dir, 'scaling_impact_barplot_HighRes.png')
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✅ Новата професионална графика е запазена в: {output_file}")