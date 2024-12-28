import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

mask_data = {
    "Mask Type": [
        "Pigment network",
        "Negative network",
        "Streaks",
        "Milia like cyst",
        "Globules",
    ],
    "Number of blank images": [1071, 2404, 2494, 1912, 1991],
    "% Blank images": [41, 93, 96, 74, 77],
    "Pixel Ratio": [0.1370, 0.0309, 0.01272, 0.0067, 0.0042],
}

data_df = pd.DataFrame(mask_data)

sns.set_theme(style="whitegrid")

plt.figure(figsize=(12, 6))
sns.barplot(
    x="Mask Type", 
    y="Number of blank images", 
    data=data_df, 
    palette="viridis"
)
plt.title("Number of Blank Images by Mask Type", fontsize=16)
plt.xlabel("Mask Type", fontsize=12)
plt.ylabel("Number of Blank Images", fontsize=12)
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

plt.figure(figsize=(12, 6))
sns.barplot(
    x="Mask Type", 
    y="% Blank images", 
    data=data_df, 
    palette="coolwarm"
)
plt.title("Percentage of Blank Images by Mask Type", fontsize=16)
plt.xlabel("Mask Type", fontsize=12)
plt.ylabel("% Blank Images", fontsize=12)
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

plt.figure(figsize=(12, 6))
sns.barplot(
    x="Mask Type", 
    y="Pixel Ratio", 
    data=data_df, 
    palette="magma"
)
plt.title("Pixel Ratio (Ones / Ones + Zeroes) by Mask Type", fontsize=16)
plt.xlabel("Mask Type", fontsize=12)
plt.ylabel("Pixel Ratio", fontsize=12)
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()
