import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("C:\\Users\\sm380923\\Desktop\\Research\\Visibility-Networks\\rewrite2\\Site_Image_Counts.csv")
df.columns = [col.strip() for col in df.columns]

bins = [0, 12, 25, 50, 75, 100, 150, 200, 300, 500, 1000]
labels = [f"{bins[i]+1}-{bins[i+1]}" for i in range(len(bins)-1)]
labels[0] = f"{bins[0]}-{bins[1]}"

df['Range'] = pd.cut(df['Number of Images'], bins=bins, labels=labels, right=True)

range_counts = df['Range'].value_counts().sort_index()
range_counts = range_counts[range_counts > 0]  # 🚫 Remove empty bins

plt.figure(figsize=(10, 6))
bars = plt.bar(
    range_counts.index,
    range_counts.values,
    color="#4caf50",                       
    edgecolor=(0.5, 0.5, 0.5, 0.3),
    linewidth=0.7,
    width=1.0
)

for bar in bars:
    height = bar.get_height()
    if height > 0:
        plt.text(bar.get_x() + bar.get_width()/2, height + 0.5, str(int(height)),
                 ha='center', va='bottom', fontsize=10)

plt.xlabel("Image Count Range per Site", fontsize=13)
plt.ylabel("Number of Sites", fontsize=13)
plt.title("Number of Sites per Image Count Range", fontsize=15)
plt.xticks(rotation=45)
plt.grid(axis='y', linestyle='--', alpha=0.6)
plt.tight_layout()
plt.show()