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

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LogisticRegression
import joblib

model = Pipeline([
    ('scaler', MinMaxScaler()),
    ('logreg', LogisticRegression(max_iter=1000))
])

model.fit(x, y)
joblib.dump(model, 'my_model.joblib')


