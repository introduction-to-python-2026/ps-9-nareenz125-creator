!wget htt!wget https://raw.githubusercontent.com/yotam-biu/ps9/main/parkinsons.csv -O /content/parkinsons.csv
!wget https://raw.githubusercontent.com/yotam-biu/python_utils/main/lab_setup_do_not_edit.py -O /content/lab_setup_do_not_edit.py

import lab_setup_do_not_edit
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LogisticRegression
import joblib

# قراءة الداتا
df = pd.read_csv('parkinsons.csv')
df = df.dropna()
print(df.columns.to_list())

# Optional visualization
sns.pairplot(df, hue='status', diag_kind='kde', corner=True)
plt.show()

# اختيار الميزتين
selected_features = ['MDVP:Flo(Hz)', 'MDVP:Jitter(%)']
x = df[selected_features]
y = df['status']

# انشاء pipeline مع scaler
model = Pipeline([
    ('scaler', MinMaxScaler()),
    ('logreg', LogisticRegression(max_iter=1000))
])

# تدريب على كل البيانات
model.fit(x, y)

# حفظ الموديل
joblib.dump(model, 'my_model.joblib')
