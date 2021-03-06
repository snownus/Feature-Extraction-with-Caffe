import matplotlib.pyplot as plt
plt.style.use('ggplot')
import numpy as np
import sklearn as skl
from sklearn.grid_search import GridSearchCV

if __name__ == '__main__':
    p1 = [0.1, 1, 10, 100]
    p2 = [1e-2, 1e-4, 1e-6, 1e-8, 1e-10]
    param_grid = {"C": p1, "gamma": p2}
    from sklearn.svm import SVC
    clf = GridSearchCV(SVC(class_weight='auto'), \
                        param_grid=param_grid, n_jobs=-1)
    features = np.load('train_features.npy')
    labels = np.load('train_labels.npy')
    clf.fit(features, labels)
    print(clf.best_estimator_)

    test_feats = np.load('test_features.npy')
    predicts = clf.predict(test_feats)
    test_labels = np.load('test_labels.npy')
    class_names = np.load('class_names.npy')
    
    accuracy = skl.metrics.accuracy_score(y_true=test_labels, y_pred=predicts)
    print 'accuracy: ' + str(accuracy)

    report = skl.metrics.classification_report( \
        y_true=test_labels, y_pred=predicts, target_names=class_names)
    print(report)

    scores = [x[1] for x in clf.grid_scores_]
    scores = np.array(scores).reshape(len(p1), len(p2))
    plt.ylabel('C')
    plt.xlabel('gamma')
    #plt.imshow(scores, interpolation='nearest', cmap='Reds')
    plt.contourf(range(len(p2)), range(len(p1)), scores, cmap='Reds')
    plt.colorbar().set_label('accuracy')
    plt.xticks(range(len(p2)), p2, rotation=45)
    plt.yticks(range(len(p1)), p1)
    plt.tight_layout()
    plt.show()
    plt.savefig('svm_gridsearch.png', transparent=True)
