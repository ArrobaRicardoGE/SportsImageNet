import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('dataset/train.csv')

sns.countplot(df, x='label')
plt.show()