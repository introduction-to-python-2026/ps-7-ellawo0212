import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import fetch_openml
import pandas as pd

wine_data = fetch_openml(name='wine', version=1, as_frame=True, parser='pandas')
wine = wine_data.frame
features = ['Alcohol', 'Color_intensity', 'Hue']

fig, axs  = plt.subplots(1, len(features), figsize = (20,3))

for ax, f in zip(axs, features):
    ax.hist(wine[f], color='skyblue', edgecolor='black')
    ax.set_xlabel(f)

reference_feature = features[1]
y = wine[reference_feature]

fig, axs  = plt.subplots(1, len(features), figsize = (20,6))

for ax, f in zip(axs, features):
  ax.scatter(wine[f], y)
  ax.set_xlabel(f)

plt.show()

reference_feature = features[0]  # The reference feature
comparison_feature = features[1]  # A feature to compare to

# Create a scatter plot for the selected pair
plt.figure(figsize=(8, 6))
plt.scatter(wine[reference_feature], wine[comparison_feature], alpha=0.6)
plt.xlabel(reference_feature)
plt.ylabel(comparison_feature)

# Save the plot as an image file
plt.savefig('correlation_plot.png')

plt.show()
