# Feature Extraction with Caffe
A simple python code of feature extraction with caffe refered to [wellflat/cat-fancier](https://github.com/wellflat/cat-fancier/tree/master/classifier)  
Including some examples classify the Oxford-IIIT Pet Dataset using scikit-learn

## Dependencies
Caffe, Python 2, NumPy, scikit-learn, matplotlib  

## Installation


## Usage
make sure that the locations (caffe_root, images, and model files) are correctly designated
```py
#feature_extract.py
caffe_root = '../'
image_dir = caffe_root + "working/oxford_pet_dataset/"
MEAN_FILE = caffe_root + 'python/caffe/imagenet/ilsvrc_2012_mean.npy'
MODEL_FILE = caffe_root + 'models/bvlc_reference_caffenet/deploy_feature.prototxt'
PRETRAINED = caffe_root + 'models/bvlc_reference_caffenet/bvlc_reference_caffenet.caffemodel'
FEAT_LAYER = 'fc6wi'
```

## Examples
* Classification of Oxford-IIIT Pet Dataset using SVM
```py
> report = sklearn.metrics.classification_report(y_true, y_pred, target_names)
> print(report)
                   precision    recall  f1-score   support

       Abyssinian       0.81      0.84      0.82        50
           Bengal       0.75      0.76      0.75        50
           Birman       0.69      0.70      0.69        50
           Bombay       0.92      0.94      0.93        50
British Shorthair       0.74      0.90      0.81        50
     Egyptian Mau       0.87      0.90      0.88        50
       Maine Coon       0.86      0.88      0.87        50
          Persian       0.88      0.74      0.80        50
          Ragdoll       0.62      0.68      0.65        50
     Russian Blue       0.93      0.78      0.85        50
          Siamese       0.79      0.66      0.72        50
           Sphynx       0.90      0.90      0.90        50

      avg / total       0.81      0.81      0.81       600
```
* Confusion matrix
![Confusion Matrix (SVM)](/examples/svm_cmatrix.png)  

* Contour of grid search (SVM)
![Grid Search (SVM)](/examples/svm_gridsearch.png)  

