# ==========================
# Индивидуальный проект
# Версия 3 (Feature Engineering)
# ==========================

import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier

# ==========================
# Загрузка данных
# ==========================

train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')

# ==========================
# Знакомство с данными
# ==========================

print("========== Первые строки train.csv ==========")
print(train.head())

print("\n========== Информация о train.csv ==========")
train.info()

print("\n========== Статистика ==========")
print(train.describe())

# ==========================
# Проверка гипотез
# ==========================

print("\n========== Гипотеза №1 ==========")
print("Влияет ли пол на покупку курса?")
print(train.groupby('sex')['result'].mean())

print("\n========== Гипотеза №2 ==========")
print("Влияет ли наличие фотографии?")
print(train.groupby('has_photo')['result'].mean())

print("\n========== Гипотеза №3 ==========")
print("Влияет ли наличие телефона?")
print(train.groupby('has_mobile')['result'].mean())

print("\n========== Гипотеза №4 ==========")
print("Влияет ли место работы или учёбы?")
print(train.groupby('occupation_type')['result'].mean())

# ==========================
# Проверка пропусков
# ==========================

print("\n========== Пропущенные значения ==========")
print(train.isnull().sum())

# ==================================================
# FEATURE ENGINEERING
# Создание новых признаков
# ==================================================

# Насколько заполнен профиль
train['profile_completed'] = train['has_photo'] + train['has_mobile']
test['profile_completed'] = test['has_photo'] + test['has_mobile']

# Есть ли информация о работе
train['has_career'] = (train['career_start'] != 'False').astype(int)
test['has_career'] = (test['career_start'] != 'False').astype(int)

# Есть ли информация об окончании обучения
train['has_graduation'] = (train['graduation'] > 0).astype(int)
test['has_graduation'] = (test['graduation'] > 0).astype(int)

# Популярный пользователь
train['popular_user'] = (train['followers_count'] >= 500).astype(int)
test['popular_user'] = (test['followers_count'] >= 500).astype(int)

print("\n========== Новые признаки ==========")

print(train[
    [
        'profile_completed',
        'has_career',
        'has_graduation',
        'popular_user'
    ]
].head())

# ==========================
# Удаление ненужных столбцов
# ==========================

columns = [
    'bdate',
    'education_form',
    'education_status',
    'langs',
    'life_main',
    'people_main',
    'city',
    'last_seen',
    'occupation_type',
    'occupation_name',
    'career_start',
    'career_end'
]

train = train.drop(columns=columns, errors='ignore')
test = test.drop(columns=columns, errors='ignore')

# ==========================
# Подготовка данных
# ==========================

X = train.drop('result', axis=1)

y = train['result']

# Сохраняем id пользователей

ID = test['id']

# id не участвует в обучении

X = X.drop('id', axis=1)
test = test.drop('id', axis=1)

# На всякий случай делаем одинаковый порядок столбцов

test = test[X.columns]

# ==========================
# Масштабирование
# ==========================

scaler = StandardScaler()

X = scaler.fit_transform(X)

test = scaler.transform(test)

# ==========================
# Создание модели
# ==========================

classifier = KNeighborsClassifier(
    n_neighbors=3
)

classifier.fit(X, y)

# ==========================
# Предсказание
# ==========================

y_pred = classifier.predict(test)

# ==========================
# Создание DataFrame
# ==========================

result = pd.DataFrame({
    'id': ID,
    'result': y_pred
})

# ==========================
# Сохранение результата
# ==========================

result.to_csv(
    'result_v3.csv',
    index=False
)

print("\n========== Результат ==========")
print("Файл result_v3.csv успешно создан!")

print("\nПервые строки файла:")
print(result.head())
