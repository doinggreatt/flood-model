import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns 
import numpy as np

df = pd.read_excel('main.xlsx')

df['month'] = df['date'].dt.month

df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
features_considered = ['water_level', 'water_cons', 'parc_press', 'osadki', 'month_sin']
print('Please, input the id of water body to see a heat map')
print('Possible options: 11661, 11129, 11143, 11146')
res_id = int(input('Your option:'))
df_f = df[df['res_id'] == res_id]
features = df_f[features_considered]
corr = features.corr()
sns.heatmap(corr, annot=True, cmap='coolwarm')
plt.show()
