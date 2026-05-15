# src/text_cleaner.py

import re
from sklearn.base import BaseEstimator, TransformerMixin

class TextCleaner(BaseEstimator, TransformerMixin):
    """
    Потребителски трансформатор за интелигентно почистване на текст.
    """
    def __init__(self, to_lower=True, replace_urls=False, replace_emails=False, replace_numbers=False, remove_punctuation=False):
        self.to_lower = to_lower
        self.replace_urls = replace_urls
        self.replace_emails = replace_emails
        self.replace_numbers = replace_numbers
        self.remove_punctuation = remove_punctuation

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        X_cleaned = []
        for text in X:
            if not isinstance(text, str):
                text = str(text) if text is not None else ""
                
            if self.to_lower:
                text = text.lower()
                
            # НОВО: Замяна на линкове (http://..., www...) с токен _URL_
            if self.replace_urls:
                text = re.sub(r'http[s]?://\S+|www\.\S+', ' _URL_ ', text)
                
            # НОВО: Замяна на имейли с токен _EMAIL_
            if self.replace_emails:
                text = re.sub(r'\S+@\S+', ' _EMAIL_ ', text)
                
            # Замяна на числа с токен _NUM_
            if self.replace_numbers:
                text = re.sub(r'\d+', ' _NUM_ ', text)
                
            # Премахване на пунктуацията (ВНИМАНИЕ: запазваме символа '_', за да не счупим токените)
            if self.remove_punctuation:
                text = re.sub(r'[^\w\s_]', '', text)
                
            # Почистване на двойни интервали
            text = re.sub(r'\s+', ' ', text).strip()
            
            X_cleaned.append(text)
            
        return X_cleaned