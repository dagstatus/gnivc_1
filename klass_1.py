import pandas as pd
import numpy as np
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# nltk.download('stopwords')
# nltk.download('punkt')

# загрузка данных
# df1 = pd.read_csv('data1.csv')
# df2 = pd.read_csv('data2.csv')
df2 = pd.read_excel('гнивц_1.xlsx')
df1 = pd.read_excel('СТП_1.xlsx')

# объединение данных для удобства предобработки текстов
df = pd.concat([df1, df2], axis=0)

# предобработка текстов
stop_words = stopwords.words('russian')
def preprocess_text(text):
    tokens = word_tokenize(text.lower())
    tokens = [token for token in tokens if token.isalpha() and token not in stop_words]
    return ' '.join(tokens)
df['text'] = df['text'].apply(preprocess_text)

# разделение данных на первый и второй датафреймы
df1 = df.iloc[:len(df1), :]
df2 = df.iloc[len(df1):, :]

# векторизация текстов
vectorizer = TfidfVectorizer()
X1 = vectorizer.fit_transform(df1['text'])
X2 = vectorizer.transform(df2['text'])

# поиск наиболее похожих текстов
similarities = cosine_similarity(X1, X2)
most_similar = np.argmax(similarities, axis=1)

# добавление найденных текстов в первый датафрейм
df1['most_similar_text'] = df2.iloc[most_similar, :]['text'].values

# сохранение результата
# df1.to_csv('result.csv', index=False)
df1.to_excel('result.xlsx')
