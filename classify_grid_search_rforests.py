import matplotlib.pyplot as plt
plt.style.use('ggplot')
import numpy as np
import sklearn as skl
from sklearn.grid_search import GridSearchCV

if __name__ == '__main__':
    p1 = [64, 128, 256, 512]
    p2 = [10, 50, 100, 500, 1000]
    param_grid = {"max_features": p1, #recommand: sqrt(n)
                  "n_estimators": p2} #number of trees
    from sklearn.ensemble import RandomForestClassifier
    clf = GridSearchCV( \
        RandomForestClassifier(), param_grid=param_grid, n_jobs=-1)
    features = np.load('train_features.npy')
    labels = np.load('train_labels.npy')
    np.random.seed(0)
    indices = np.random.permutation(len(labels))
    features = features[indices[:300]]
    labels = labels[indices[:300]]
    clf.fit(features, labels)
    print(clf.best_estimator_)

    test_feats = np.load('test_features.npy')
    predicts = clf.predict(test_feats)
    test_labels = np.load('test_labels.npy')
    class_names = np.load('class_names.npy')
    report = skl.metrics.classification_report( \
        y_true=test_labels, y_pred=predicts, target_names=class_names)
    print(report)

    scores = [x[1] for x in clf.grid_scores_]
    scores = np.array(scores).reshape(len(p1), len(p2))
    plt.ylabel('max features')
    plt.xlabel('number of estimators')
    #plt.imshow(scores, interpolation='nearest', cmap='Reds')
    plt.contourf(range(len(p2)), range(len(p1)), scores, cmap='Reds')
    plt.colorbar().set_label('accuracy')
    plt.xticks(range(len(p2)), p2, rotation=45)
    plt.yticks(range(len(p1)), p1)
    plt.tight_layout()
    plt.show()
    plt.savefig('rforests_gridsearch.png', transparent=True)
