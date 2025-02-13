import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('results.csv', index_col=0)
print(f"Dataframe shape: {df.shape}")
fig, axes = plt.subplots(2, 2, figsize=(10, 5))
axes = axes.flatten()
indexes_to_see = []
colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple', 'tab:brown', 'tab:pink', 'tab:gray',
          'tab:olive', 'tab:cyan']
for col, ax in zip(df.columns, axes):
    data = df[col].dropna()
    data.plot(kind='hist', bins=20, ax=ax)
    ax.set_title(f'{col}')
    for i in range(0, 17):
        ax.axvline(data.iloc[i], linestyle='--', linewidth=1, label=data.index[i], c=colors[i % 10])
        if col == 'ProtBert_MoLFormer':
            indexes_to_see.append(data.index[i])
    # ax.legend()
fig.tight_layout()
plt.show()
all_res = []
for i, row in df.head(18).iterrows():
    mol_values = []
    for col in df.columns:
        larger_elements = df[df[col] > row[col]]
        all_res.append(
            f"{i:<10} {col:<20} {row[col]:<7.3f} {len(larger_elements):<10,} {len(larger_elements) / len(df):<10.2%}")
        mol_values.append(len(larger_elements) / len(df))
    print(f"{i:<10} {np.median(mol_values):<7.3f}")
    # print(f"{i:<10} {col:<20} {row[col]:<7.3f} {len(larger_elements):<10,} {len(larger_elements)/len(df):<10.2%}")
all_res = sorted(all_res, key=lambda x: float(x.split()[-2].replace(",", "")), reverse=False)
print("\n".join(all_res))

from papers_count import get_google_scholar_results

res = df['ProtBert_MoLFormer'].sort_values(ascending=False)
print(res.head(10))
for i in res.index[:10]:
    print(i, get_google_scholar_results(i))
