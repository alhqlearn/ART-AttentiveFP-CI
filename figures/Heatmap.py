import matplotlib.pyplot as plt
import numpy as np

# Data values provided
values = [
    93.82564, 99.51649, 98.41206, 94.0268, 89.99725,
    88.90441, 85.75517, 92.23479, 90.1359, 57.750267,
    100.51225, 87.004745, 93.350006, 98.164246, 88.7362,
    90.59686, 84.881645, 86.9066, 90.194916, 87.29776,
    94.79737, 97.10336, 87.55383, 99.75165, 95.163445,
    97.62736, 95.53287, 95.8561, 88.97369, 91.04035,
    102.20187, 97.10336, 102.19215, 86.07272, 94.780304
]

# Convert to numpy array and reshape to 5x7
data = np.array(values).reshape(5, 7)

# Plot heatmap
plt.figure(figsize=(8, 4))
heatmap = plt.imshow(data, cmap="viridis", aspect="auto")

# Add colorbar on the left side
cbar = plt.colorbar(heatmap, label="Value", location="left")

# Normalize to [0,1] to check brightness
norm = plt.Normalize(vmin=data.min(), vmax=data.max())
cmap = plt.get_cmap("viridis")

# Add annotations with dynamic color
for i in range(data.shape[0]):
    for j in range(data.shape[1]):
        val = data[i, j]
        rgba = cmap(norm(val))
        brightness = 0.299*rgba[0] + 0.587*rgba[1] + 0.114*rgba[2]
        text_color = "black" if brightness > 0.5 else "white"
        plt.text(j, i, f"{val:.0f}", ha='center', va='center',
                 color=text_color, fontsize=12)

# Titles and labels
plt.title("7x5 Heatmap of Values")
plt.xlabel("Column Index")
plt.ylabel("Row Index")
# Increase colorbar text size
cbar.ax.tick_params(labelsize=12)       # numbers on colorbar
cbar.set_label("Value", fontsize=14)    # the label "Value"

# Save figure
plt.savefig("heatmap.png", dpi=600, bbox_inches="tight")
plt.show()
