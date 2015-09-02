import matplotlib.pyplot as plt
plt.style.use('ggplot')
import numpy as np
import sklearn as skl

from sklearn.svm import SVC
clf = SVC(C=10.0, gamma=1e-6, class_weight='auto')
features = np.load('train_features.npy')
labels = np.load('train_labels.npy')
clf.fit(features, labels)
print(clf)

test_feats = np.load('test_features.npy')
predicts = clf.predict(test_feats)
test_labels = np.load('test_labels.npy')
class_names = np.load('class_names.npy')

accuracy = skl.metrics.accuracy_score(y_true=test_labels, y_pred=predicts)
print 'accuracy: ' + str(accuracy)

report = skl.metrics.classification_report( \
    y_true=test_labels, y_pred=predicts, target_names=class_names)
print(report)

cmatrix = skl.metrics.confusion_matrix(y_true=test_labels, y_pred=predicts)
print(cmatrix)

plt.imshow(cmatrix, interpolation='none', cmap='Reds')
plt.xlabel('Predicted label')
plt.ylabel('True label')
plt.colorbar()
plt.xticks(range(len(class_names)), range(len(class_names)), rotation=45)
plt.yticks(range(len(class_names)), range(len(class_names)))
plt.grid(False)
plt.tight_layout()
plt.show()
plt.savefig('svm_cmatrix.png', transparent=True)
