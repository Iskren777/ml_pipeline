# src/pipeline_builder.py

import yaml
from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler, PowerTransformer, OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from src.text_cleaner import TextCleaner # Вашият custom transformer

# ---------------------------------------------------------
# Речници за картографиране (Mapping): 
# Превеждат текст от YAML към реални scikit-learn класове
# ---------------------------------------------------------
SCALER_MAP = {
    "StandardScaler": StandardScaler(),
    "MinMaxScaler": MinMaxScaler(),
    "RobustScaler": RobustScaler(),             # НОВО: Игнорира екстремни стойности (outliers)
    "PowerTransformer": PowerTransformer(),     # НОВО: Прави данните симетрични (Gaussian-like)
    "None": "passthrough"
}

MODEL_MAP = {
    "LogisticRegression": LogisticRegression(max_iter=100), # Връщаме на 100 итерации, за да видим кога се проваля без скалиране
    "RandomForest": RandomForestClassifier(random_state=42),
    "KNN": KNeighborsClassifier() # НОВО: Модел, базиран на разстояния
}

def load_config(config_path: str) -> dict:
    """Зарежда YAML конфигурационен файл."""
    with open(config_path, 'r', encoding='utf-8') as file:
        return yaml.safe_load(file)

def build_dynamic_pipeline_and_grid(config: dict):
    """
    Изгражда базов Pipeline и динамична решетка от параметри (param_grid)
    на базата на подадената конфигурация.
    """
    
    # 1. Изграждане на Preprocessor-а (Трансформациите)
    num_features = config['data']['numeric_features']
    cat_features = config['data']['categorical_features']

    # Дефинираме стъпките за числовите данни. 
    # Слагаме 'passthrough' като временен заместител (placeholder) за скалирането.
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', 'passthrough') 
    ])

    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])

    preprocessor = ColumnTransformer(transformers=[
        ('num', numeric_transformer, num_features),
        ('cat', categorical_transformer, cat_features)
    ])

    # 2. Създаване на Базовия Pipeline
    # Слагаме LogisticRegression като placeholder. GridSearchCV ще го подменя.
    base_pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('model', LogisticRegression()) 
    ])

    # 3. Изграждане на динамичния param_grid за експериментите
    scalers_from_config = config['transformations']['scalers_to_test']
    actual_scalers = [SCALER_MAP[s] for s in scalers_from_config]
    
    param_grid = []
    
    # Итерираме през всеки модел, дефиниран в YAML файла
    for model_info in config['models']:
        model_name = model_info['name']
        model_instance = MODEL_MAP[model_name]
        
        # Създаваме речник с настройки за конкретния модел
        grid_params = {
            'model': [model_instance],
            'preprocessor__num__scaler': actual_scalers
        }
        
        # Добавяме специфичните хиперпараметри за модела
        if 'params' in model_info:
            for param_name, param_values in model_info['params'].items():
                # Проверяваме дали параметърът вече има префикс. Ако няма, го добавяме.
                if not param_name.startswith('model__'):
                    grid_key = f"model__{param_name}"
                else:
                    grid_key = param_name
                grid_params[grid_key] = param_values
            
        param_grid.append(grid_params)

    return base_pipeline, param_grid

def build_nlp_pipeline_and_grid(config: dict):
    """
    Изгражда Pipeline специално за обработка на естествен език (NLP).
    """
    # 1. Базов Pipeline за текст
    base_pipeline = Pipeline([
        ('cleaner', TextCleaner(to_lower=True)), # Стъпка 1: Нашето почистване
        ('vectorizer', TfidfVectorizer(max_features=3000)), # Стъпка 2: Думи към числа
        ('model', LogisticRegression()) # Стъпка 3: Модел (ще се подменя)
    ])
    
    # 2. Изграждане на решетката с параметри (Param Grid)
    param_grid = []
    for model_info in config['models']:
        model_name = model_info['name']
        model_instance = MODEL_MAP[model_name] # Вземаме модела от речника
        
        grid_params = {'model': [model_instance]}
        if 'params' in model_info:
            grid_params.update(model_info['params'])
            
        param_grid.append(grid_params)
        
    return base_pipeline, param_grid

# ==========================================
# Тест на системата
# ==========================================
if __name__ == "__main__":
    # Симулираме четене на конфигурацията
    config = load_config("C:\\Users\\kottk\\Desktop\\Много важно\\diploma_ml_pipeline\\configs\\experiment_1.yaml")
    
    pipeline, grid = build_dynamic_pipeline_and_grid(config)
    
    print("=== Базов Pipeline ===")
    print(pipeline)
    
    print("\n=== Генерирана решетка за експерименти (Param Grid) ===")
    import pprint
    pprint.pprint(grid)