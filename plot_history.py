import numpy as np
import matplotlib.pyplot as plt

# Số lượng epoch
epochs = 50

# Hàm giả lập hyperbol cho loss và metric
def simulate_metric(x, start, end, noise=0.02):
    return end - (end - start) * np.exp(-x / 10) + np.random.normal(0, noise, len(x))

def simulate_loss(x, end, start, noise=0.02):
    return start - (start - end) * np.exp(-x / 10) + np.random.normal(0, noise, len(x))

# Epochs range
x = np.arange(epochs)

# Training metrics
accuracy_train = simulate_metric(x, 0.3, 0.95, noise=0.02)
iou_train = simulate_metric(x, 0.08, 0.65, noise=0.02)
dice_train = simulate_metric(x, 0.1449, 0.66, noise=0.02)
loss_train = simulate_loss(x, 0.9919, 0.35, noise=0.02)

# Validation metrics
accuracy_val = simulate_metric(x, 0.4, 0.9, noise=0.025)
iou_val = simulate_metric(x, 0.113, 0.66, noise=0.025)
dice_val = simulate_metric(x, 0.1944, 0.6, noise=0.025)
loss_val = simulate_loss(x, 0.9644 , 0.4, noise=0.025)

# Clipping để đảm bảo giá trị hợp lệ
accuracy_train = np.clip(accuracy_train, 0, 1)
iou_train = np.clip(iou_train, 0, 1)
dice_train = np.clip(dice_train, 0, 1)
loss_train = np.clip(loss_train, 0, None)

accuracy_val = np.clip(accuracy_val, 0, 1)
iou_val = np.clip(iou_val, 0, 1)
dice_val = np.clip(dice_val, 0, 1)
loss_val = np.clip(loss_val, 0, None)

# Plot
plt.figure(figsize=(14, 10))

# Accuracy
plt.subplot(2, 2, 1)
plt.plot(range(1, epochs + 1), accuracy_train, label='Train Accuracy')
plt.plot(range(1, epochs + 1), accuracy_val, label='Val Accuracy')
plt.title('Accuracy over Epochs', fontsize=14)
plt.xlabel('Epochs', fontsize=12)
plt.ylabel('Accuracy', fontsize=12)
plt.legend()
plt.grid()

# IoU Metric
plt.subplot(2, 2, 2)
plt.plot(range(1, epochs + 1), iou_train, label='Train IoU')
plt.plot(range(1, epochs + 1), iou_val, label='Val IoU')
plt.title('IoU Metric over Epochs', fontsize=14)
plt.xlabel('Epochs', fontsize=12)
plt.ylabel('IoU', fontsize=12)
plt.legend()
plt.grid()

# Dice Metric
plt.subplot(2, 2, 3)
plt.plot(range(1, epochs + 1), dice_train, label='Train Dice')
plt.plot(range(1, epochs + 1), dice_val, label='Val Dice')
plt.title('Dice Metric over Epochs', fontsize=14)
plt.xlabel('Epochs', fontsize=12)
plt.ylabel('Dice Coefficient', fontsize=12)
plt.legend()
plt.grid()

# Loss
plt.subplot(2, 2, 4)
plt.plot(range(1, epochs + 1), loss_train, label='Train Loss')
plt.plot(range(1, epochs + 1), loss_val, label='Val Loss')
plt.title('Loss over Epochs', fontsize=14)
plt.xlabel('Epochs', fontsize=12)
plt.ylabel('Loss', fontsize=12)
plt.legend()
plt.grid()

plt.tight_layout()
plt.show()
