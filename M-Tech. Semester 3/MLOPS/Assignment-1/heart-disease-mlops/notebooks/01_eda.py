import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

df = pd.read_csv("../data/processed/heart_clean.csv")
df.head()

sns.countplot(x="target", data=df)
plt.show()
