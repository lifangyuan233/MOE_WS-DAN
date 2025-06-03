import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, roc_auc_score, f1_score

# 示例数据（请替换为你的tp, tn, fp, fn值）
total1 = 1000
total2 = 17
tp = 3
tn = 656
fp = total1-tn
fn = total2 - tp

# 计算混淆矩阵
cm = np.array([[tn, fp], [fn, tp]])

# 生成混淆矩阵图
plt.figure(figsize=(6, 4))
plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
plt.title('Confusion Matrix')
plt.colorbar()
tick_marks = np.arange(2)
plt.xticks(tick_marks, ['0', '1'])
plt.yticks(tick_marks, ['0', '1'])

# 在混淆矩阵上添加文本标签
thresh = cm.max() / 2
for i, j in np.ndindex(cm.shape):
    plt.text(j, i, cm[i, j], horizontalalignment='center',
             color='white' if cm[i, j] > thresh else 'black')

plt.ylabel('True label')
plt.xlabel('Predicted label')
plt.tight_layout()

# 保存混淆矩阵图
plt.savefig('confusion_matrix.png')
plt.show()

# 计算相关指标
ACC = (tn + tp) / (tn + tp + fn + fp)
SPE = tn / (tn + fp)  # 特异度
SEN = tp / (tp + fn)  # 灵敏度
PRE = precision_score([0] * tn + [1] * tp + [0] * fp + [1] * fn, [0] * (tn + fp) + [1] * (tp + fn))
AUC = roc_auc_score([0] * tn + [1] * tp + [0] * fp + [1] * fn, [0] * (tn + fp) + [1] * (tp + fn))  # 示例AUC计算
f1 = f1_score([0] * tn + [1] * tp + [0] * fp + [1] * fn, [0] * (tn + fp) + [1] * (tp + fn))

# 输出结果
print('TN:', tn, 'TP:', tp, 'FP:', fp, 'FN:', fn)
print('ACC:', ACC, 'SPE:', SPE, 'SEN:', SEN, 'PRE:', PRE, 'AUC:', AUC, 'F1:', f1)
