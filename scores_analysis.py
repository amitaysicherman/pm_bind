import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('results.csv', index_col=0)
print(f"Dataframe shape: {df.shape}")
fig,axes = plt.subplots(2, 2, figsize=(10,5))
axes = axes.flatten()
colors=['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple', 'tab:brown', 'tab:pink', 'tab:gray', 'tab:olive', 'tab:cyan']
for col,ax in zip(df.columns,axes):
    data=df[col].dropna()
    data.plot(kind='hist', bins=20,ax=ax)
    ax.set_title(f'{col}')
    for i in range(0, 17):
        ax.axvline(data.iloc[i], linestyle='--', linewidth=1, label=data.index[i],c=colors[i%10])
    # ax.legend()
fig.tight_layout()
plt.show()
