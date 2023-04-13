import pandas as pd
from gensim.models import Word2Vec
from gensim.utils import simple_preprocess
import nltk
from nltk.corpus import stopwords
from pymorphy2 import MorphAnalyzer
import re
import cupy as cp
import spacy

# Инициализируем объект лемматизатора
morph = MorphAnalyzer()

# Загружаем список стоп-слов для русского языка
nltk.download('stopwords')
stop_words = set(stopwords.words('russian'))


# def preprocess_text(text):
#     # Приводим текст к нижнему регистру
#     text = text.lower()
#
#     # Удаляем все символы кроме букв и цифр
#     text = re.sub(r'[^a-zA-Zа-яА-Я0-9]', ' ', text)
#
#     # Разбиваем текст на слова
#     words = text.split()
#
#     # Лемматизируем каждое слово
#     words = [morph.parse(w)[0].normal_form for w in words]
#
#     # Удаляем стоп-слова
#     words = [w for w in words if not w in stop_words]
#
#     # Склеиваем слова обратно в текст
#     text = ' '.join(words)
#
#     return text
#

nlp = spacy.load('ru_core_news_sm')
def preprocess_text(text):
    doc = nlp(text)
    lemmas = [token.lemma_.lower() for token in doc if not token.is_stop and token.lemma_.isalpha()]
    lemmas_arr = cp.array(lemmas)

    return lemmas_arr


# Создаем первый датафрейм
# df1 = pd.DataFrame({'text': ['красная кошка на окне', 'синяя машина на улице', 'зеленое дерево в парке']})

# Создаем второй датафрейм
# df2 = pd.DataFrame({'text': ['кошка на окне', 'машина на улице', 'дерево в парке']})

df1 = pd.read_excel('гнивц_1.xlsx')
df2 = pd.read_excel('СТП_1.xlsx')

print("Делаем предобработку текста")
df1['text'] = df1['text'].apply(preprocess_text)
df2['text'] = df2['text'].apply(preprocess_text)

# Объединяем тексты из двух датафреймов в один список
texts = list(df1['text']) + list(df2['text'])

print("Делаем токены")
# Преобразуем тексты в список токенов
tokens = [simple_preprocess(text) for text in texts]

# Создаем модель Word2Vec
print("Начинаем обучение модели")
model = Word2Vec(tokens, window=5, min_count=1, workers=4)
model.wv.save_word2vec_format('model.bin', binary=True)

#load model
# model = Word2Vec.load('model.bin')

print("Начинаем перебор")
# Находим наиболее похожий текст из df1 для каждой строки df2
similarities = []
for text2 in df2['text']:
    text2_tokens = simple_preprocess(text2)
    text2_vec = sum([model.wv[token] for token in text2_tokens]) / len(text2_tokens)
    max_similarity = 0
    for text1 in df1['text']:
        text1_tokens = simple_preprocess(text1)
        text1_vec = sum([model.wv[token] for token in text1_tokens]) / len(text1_tokens)
        similarity = model.wv.cosine_similarities(text2_vec, [text1_vec])[0]
        if similarity > max_similarity:
            max_similarity = similarity
            most_similar_text = text1
    similarities.append(max_similarity)
    df2.loc[df2['text'] == text2, 'most_similar_text'] = most_similar_text

# Добавляем столбец с коэффициентами похожести
df2['similarity'] = similarities

df2.to_excel('res2.xlsx')
