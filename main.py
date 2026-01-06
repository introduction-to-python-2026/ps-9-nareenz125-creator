!wget https://raw.githubusercontent.com/yotam-biu/ps9/main/parkinsons.csv -O /content/parkinsons.csv
!wget https://raw.githubusercontent.com/yotam-biu/python_utils/main/lab_setup_do_not_edit.py -O /content/lab_setup_do_not_edit.py
import lab_setup_do_not_edit

import pandas as pd
df = pd.read_csv('parkinsons.csv')
df = df.dropna()
df.head()

print(df.columns.to_list())

import matplotlib.pyplot as plt
import seaborn as sns
sns.pairplot(df, hue='status', diag_kind='kde', corner=True)
plt.show()

selected_features = ['MDVP:Flo(Hz)', 'MDVP:Jitter(%)']
x = df[selected_features]
y = df['status']

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
x = scaler.fit_transform(x)

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

from sklearn.linear_model import LogisticRegression
model = LogisticRegression(max_iter = 99)
model.fit(x_train, y_train)

from sklearn.metrics import accuracy_score
y_pred = model.predict(x_test)
accuracy = accuracy_score(y_test, y_pred)
print(accuracy)

import joblib

joblib.dump(model, 'my_model.joblib')
