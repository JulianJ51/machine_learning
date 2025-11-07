import matplotlib.pyplot as plt
import numpy as np 
from matplotlib.ticker import MaxNLocator

'''

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

'''

#Performance metrics bar charts

perceptrons = ("0", "1", "2", "3", "4", "5", "6", "7", "8", "9")
balanced_error = {
    'Before': (0.685556, 0.457778, 0.532778, 0.506111, 0.450556, 0.37, 0.361667, 0.523333, 0.466667, 0.441111),
    'After': (0.0238889, 0.025, 0.0866667, 0.0794444, 0.121667, 0.107778, 0.0461111, 0.0783333, 0.0788889, 0.11),
}
precision = {
    'Before': (0.0562827, 0.146341, 0.085213, 0.0979284, 0.116451, 0.219388, 0.138249, 0.0851064, 0.107595, 0.118467),
    'After': (0.858407, 0.914286, 0.875, 0.669173, 0.865169, 0.778846, 0.764228, 0.783784, 0.674242, 0.648438),
}
recall = {
    'Before': (0.43, 0.24, 0.34, 0.52, 0.63, 0.43, 0.9, 0.24, 0.85, 0.68),
    'After': (0.97, 0.96, 0.84, 0.89, 0.77, 0.81, 0.94, 0.87, 0.89, 0.83),
}
f1_score = {
    'Before': (0.099537, 0.181818, 0.136273, 0.164818, 0.196568, 0.290541, 0.23968, 0.125654, 0.191011, 0.20178),
    'After': (0.910698, 0.936585, 0.857143, 0.763948, 0.814815, 0.794118, 0.843049, 0.824645, 0.767241, 0.72807),
}

x = np.arange(len(perceptrons))
width = 0.4
multiplier = 0

fig, ax = plt.subplots(layout='constrained', figsize=(10,4))

for attribute, measurement in balanced_error.items():
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

ax.set_ylabel('Balanced Error')
ax.set_title('Balanced Error for each Perceptron Before/After')
ax.set_xticks(x + width/2, perceptrons)
ax.legend(loc='upper left', ncols=2)
ax.set_ylim(0, 1)

plt.show()

fig, ax = plt.subplots(layout='constrained', figsize=(10,4))
multiplier = 0

for attribute, measurement in precision.items():
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

ax.set_ylabel('Precision')
ax.set_title('Precision for each Perceptron Before/After')
ax.set_xticks(x + width/2, perceptrons)
ax.legend(loc='upper left', ncols=2)
ax.set_ylim(0, 1)

plt.show()

fig, ax = plt.subplots(layout='constrained', figsize=(10,4))
multiplier = 0

for attribute, measurement in recall.items():
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

ax.set_ylabel('Recall')
ax.set_title('Recall for each Perceptron Before/After')
ax.set_xticks(x + width/2, perceptrons)
ax.legend(loc='upper left', ncols=2)
ax.set_ylim(0, 1)

plt.show()

fig, ax = plt.subplots(layout='constrained', figsize=(10,4))
multiplier = 0

for attribute, measurement in f1_score.items():
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

ax.set_ylabel('F1 Score')
ax.set_title('F1 Score for each Perceptron Before/After')
ax.set_xticks(x + width/2, perceptrons)
ax.legend(loc='upper left', ncols=2)
ax.set_ylim(0, 1)

plt.show()

#errors during training for 0-9 perceptrons
x = np.arange(1,51)
y0 = np.loadtxt("../0error.txt")
y1 = np.loadtxt("../1error.txt")
y2 = np.loadtxt("../2error.txt")
y3 = np.loadtxt("../3error.txt")
y4 = np.loadtxt("../4error.txt")
y5 = np.loadtxt("../5error.txt")
y6 = np.loadtxt("../6error.txt")
y7 = np.loadtxt("../7error.txt")
y8 = np.loadtxt("../8error.txt")
y9 = np.loadtxt("../9error.txt")

ys = [y0, y1, y2, y3, y4, y5, y6, y7, y8, y9]

for i, y in enumerate(ys):
    plt.scatter(x, y)
    plt.plot(x, y, label=f'Training Curve {i}')

plt.xlabel('Epoch')
plt.ylabel('Balanced Error')
plt.title('Training Curves for each Perceptron')
plt.legend()

plt.grid()
plt.tight_layout()
plt.show()

#heat map plots


for x in range(10):
    heat_maps_combined = np.loadtxt(f"../heatmap{x}.txt")
    heat_maps = np.array_split(heat_maps_combined, 2)

    fig, ax = plt.subplots()
    im = ax.imshow(heat_maps[0], cmap = 'viridis')
    cbar = fig.colorbar(im, ax=ax) 
    cbar.set_label("Weight Value")   

    ax.set_title(f"Weights Before Training on the {x} Perceptron")
    fig.tight_layout()
    plt.show()

    fig, ax = plt.subplots()
    im = ax.imshow(heat_maps[1], cmap='viridis')
    cbar = fig.colorbar(im, ax=ax)  
    cbar.set_label("Weight Value")  

    ax.set_title(f"Weights After Training on the {x} Perceptron")
    fig.tight_layout()
    plt.show()