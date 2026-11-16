import warnings
warnings.filterwarnings('ignore') # Отключаем все красные предупреждения, они из за модели а не из за ошибки
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

# генерация данных
print("Генерация тестовых данных")
np.random.seed(42)
n = 1000
data = {
    'Device_id': np.random.randint(1, 500, n),
    'Device_type': np.random.choice(['mobile', 'desktop'], n),
    'Datetime': pd.date_range(start='2023-07-11', periods=n, freq='h'),
    'Ev_type': np.random.choice([0, 1], n, p=[0.8, 0.2]), # 0 - показ, 1 - клик
    'Adv_type': np.random.choice(['banner', 'video'], n),
    'Region_id': np.random.choice([1, 2], n),
    'Age': np.random.randint(18, 65, n),
    'Gender': np.random.choice([1, 2], n)
}
df = pd.DataFrame(data)
df['Datetime'] = pd.to_datetime(df['Datetime'])
df['hour'] = df['Datetime'].dt.hour
print(f"Данные готовы. Размер таблицы: {df.shape}\n")

# анализ
print(" Анализ данных ")
ctr_overall = df['Ev_type'].mean()
print(f"Средний CTR: {ctr_overall:.2%}")

# CTR по часам
plt.figure(figsize=(10, 5))
ctr_by_hour = df.groupby('hour')['Ev_type'].mean()
sns.barplot(x=ctr_by_hour.index, y=ctr_by_hour.values, color='skyblue')
plt.title('CTR в зависимости от часа')
plt.xlabel('Час дня')
plt.ylabel('CTR')
plt.savefig('figures/ctr_by_hour.png')
print("Сохранен график: figures/ctr_by_hour.png")

#  CTR по устройствам
plt.figure(figsize=(6, 4))
ctr_by_device = df.groupby('Device_type')['Ev_type'].mean()
sns.barplot(x=ctr_by_device.index, y=ctr_by_device.values, palette='pastel')
plt.title('CTR по типам устройств')
plt.ylabel('CTR')
plt.savefig('figures/ctr_by_device.png')
print("Сохранен график: figures/ctr_by_device.png")

# обучение
print("\nОбучение модели")
df_ml = pd.get_dummies(df, columns=['Device_type', 'Adv_type'], drop_first=True)
X = df_ml[['Age', 'Gender', 'Region_id', 'hour', 'Device_type_mobile', 'Adv_type_video']]
y = df_ml['Ev_type']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

print(f"Точность модели: {accuracy_score(y_test, y_pred):.2f}")
print("\nОтчет по модели:\n", classification_report(y_test, y_pred))

# График 3
plt.figure(figsize=(8, 5))
importance = pd.Series(model.coef_[0], index=X.columns).sort_values()
importance.plot(kind='barh', color='lightgreen')
plt.title('Важность признаков для предсказания клика')
plt.savefig('figures/feature_importance.png')
print("Сохранен график: figures/feature_importance.png")
