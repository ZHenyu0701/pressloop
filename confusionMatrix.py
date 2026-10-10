import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

cm = np.array([[63062, 3819],
               [9930, 26361]])

plt.figure(figsize=(5,4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')

plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")

plt.show()