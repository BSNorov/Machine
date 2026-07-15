# ==========================
# Индивидуальный проект
# Версия 2 (Feature Engineering)
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

# =====================================================
# Feature Engineering
# Создаем новый признак profile_completed
# =====================================================

train['profile_completed'] = train['has_photo'] + train['has_mobile']
test['profile_completed'] = test['has_photo'] + test['has_mobile']

print("\n========== Новый признак profile_completed ==========")
print(train[['has_photo', 'has_mobile', 'profile_completed']].head())

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

# Признаки
X = train.drop('result', axis=1)

# Правильные ответы
y = train['result']

# Сохраняем ID пользователей
ID = test['id']

# ID не участвует в обучении модели
X = X.drop('id', axis=1)
test = test.drop('id', axis=1)

# ==========================
# Масштабирование данных
# ==========================

scaler = StandardScaler()

X = scaler.fit_transform(X)
test = scaler.transform(test)

# ==========================
# Создание модели KNN
# ==========================

classifier = KNeighborsClassifier(n_neighbors=3)

classifier.fit(X, y)

# ==========================
# Предсказание результатов
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

result.to_csv('result_v2.csv', index=False)

print("\n========== Результат ==========")
print("Файл result_v2.csv успешно создан!")

print("\nПервые строки файла:")
print(result.head())
