import matplotlib.pyplot as plt
import numpy as np 
from matplotlib.ticker import MaxNLocator

#Error Fraction During Training Plot

x = np.array([1,2,3])
y = np.array([0.00625,0.00625,0.00375]) #input correct values later

plt.scatter(x,y, label='Error at Epoch')
plt.plot(x,y,label='Training Curve')

plt.xlabel("Epoch Number")
plt.ylabel("Training Error Fraction")
plt.title("Error Fraction During Training")

ax = plt.gca()
ax.xaxis.set_major_locator(MaxNLocator(integer=True))

plt.legend()
plt.grid(True)
plt.show()

#Performance Bar Charts Before/After
#replace with correct values later
metrics = ("Error Fraction", "Precision", "Recall", "F1 Score")
performance = {
    'Before': (0.595, 0.268293, 0.11, 0.156028),
    'After': (0.005, 1, 0.99, 0.994975),
}

x = np.arange(len(metrics))
width = 0.3
multiplier = 0

fig, ax = plt.subplots(layout='constrained')

for attribute, measurement in performance.items():
    offset = width * multiplier
    rects = ax.bar(x + offset, measurement, width, label=attribute)
    
    labels = []
    for rect, val in zip(rects, measurement):
        if val > 0.85:  
            labels.append(f"{val:.2f}")
            rect.set_height(val)  
            ax.text(rect.get_x() + rect.get_width()/2, val - 0.05, f"{val:.2f}", 
                    ha='center', va='top', color='white', fontweight='bold')
        else:
            labels.append(f"{val:.2f}")
            ax.text(rect.get_x() + rect.get_width()/2, val + 0.01, f"{val:.2f}", 
                    ha='center', va='bottom', fontweight='bold')
    
    multiplier += 1

ax.set_ylabel('Performance')
ax.set_title('Performance of Perceptron')
ax.set_xticks(x + width/2, metrics)
ax.legend(loc='upper left', ncols=2)
ax.set_ylim(0, 1)

plt.show()

#Heat map plots
heat_maps_combined = np.loadtxt("../heatmap.txt")
heat_maps = np.array_split(heat_maps_combined, 2)

fig, ax = plt.subplots()
im = ax.imshow(heat_maps[0], cmap = 'viridis')
cbar = fig.colorbar(im, ax=ax) 
cbar.set_label("Weight Value")   

ax.set_title("Weights Before Training")
fig.tight_layout()
plt.show()

fig, ax = plt.subplots()
im = ax.imshow(heat_maps[1], cmap='viridis')
cbar = fig.colorbar(im, ax=ax)  
cbar.set_label("Weight Value")  

ax.set_title("Weights After Training")
fig.tight_layout()
plt.show()

