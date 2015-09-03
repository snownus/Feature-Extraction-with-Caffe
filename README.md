# Feature Extraction with Caffe
A simple python code of feature extraction with caffe refered to [wellflat/cat-fancier](https://github.com/wellflat/cat-fancier/tree/master/classifier)  
Including some examples classify the [Oxford-IIIT Pet Dataset](http://www.robots.ox.ac.uk/~vgg/data/pets/) using scikit-learn

## Dependencies
Caffe, Python 2, NumPy, scikit-learn, matplotlib  

## Installation
Download the [CaffeNet modelfile](https://github.com/BVLC/caffe/tree/master/models/bvlc_reference_caffenet)
to caffe/models/bvlc_reference_caffenet/ and other dependent files by this script
```shellsession
$ ./data/ilsvrc12/get_ilsvrc_aux.sh
```
Modify the deploy.prototxt file as following
```shellsession
$ cp models/bvlc_reference_caffenet/deploy.prototxt  
      models/bvlc_reference_caffenet/deploy_feature.prototxt
```
* deploy_feature.prototxt
```
# line 152
layer {
  name: "fc6"
  type: "InnerProduct"
  bottom: "pool5"
  # top: "fc6"
  top: "fc6wi"
  inner_product_param {
    num_output: 4096
  }
}
layer {
  name: "relu6"
  type: "ReLU"
  # bottom: "fc6"
  bottom: "fc6wi"
  top: "fc6"
}
```

## Usage  
Make sure that the following locations (caffe_root, images, and model files) are correctly designated
* feature_extract.py
```py
caffe_root = '../'
image_dir = caffe_root + "working/The Oxford-IIIT Pet Dataset/"
MEAN_FILE = caffe_root + 'python/caffe/imagenet/ilsvrc_2012_mean.npy'
MODEL_FILE = caffe_root + 'models/bvlc_reference_caffenet/deploy_feature.prototxt'
PRETRAINED = caffe_root + 'models/bvlc_reference_caffenet/bvlc_reference_caffenet.caffemodel'
FEAT_LAYER = 'fc6wi'
```
You should prepare a '.npy' file contains image filenames in numpy.array  
```shellsession
$ python feature_extract.py -i image_filenames.npy -o extracted_features.npy
```

## Examples  
* Contour of grid search (SVM)
![Grid Search (SVM)](/examples/svm_gridsearch.png)  

* Classification of Oxford-IIIT Pet Dataset using SVM  
```py
> accuracy = skl.metrics.accuracy_score(test_labels, predicts)
> print(accuracy)
0.829838709677  

> report = sklearn.metrics.classification_report(test_labels, predicts, target_names)
> print(report)
                            precision    recall  f1-score   support

          american bulldog       0.66      0.80      0.72        50
 american pit bull terrier       0.66      0.58      0.62        50
              basset hound       0.78      0.80      0.79        50
                    beagle       0.71      0.60      0.65        50
                     boxer       0.71      0.78      0.74        50
                 chihuahua       0.89      0.78      0.83        50
    english cocker spaniel       0.81      0.84      0.82        50
            english setter       0.85      0.78      0.81        50
        german shorthaired       0.95      0.82      0.88        50
            great pyrenees       0.84      0.84      0.84        50
                  havanese       0.88      0.84      0.86        50
             japanese chin       0.93      0.86      0.90        50
                  keeshond       0.98      0.98      0.98        50
                leonberger       0.88      0.92      0.90        50
        miniature pinscher       0.87      0.82      0.85        50
              newfoundland       0.85      0.92      0.88        50
                pomeranian       0.95      0.78      0.86        50
                       pug       0.96      0.98      0.97        50
             saint bernard       0.81      0.86      0.83        50
                   samoyed       0.84      0.94      0.89        50
          scottish terrier       0.90      0.90      0.90        49
                 shiba inu       0.79      0.88      0.83        50
staffordshire bull terrier       0.56      0.66      0.61        41
           wheaten terrier       0.79      0.84      0.82        50
         yorkshire terrier       0.96      0.92      0.94        50

               avg / total       0.83      0.83      0.83      1240
```
* Confusion matrix  
![Confusion Matrix (SVM)](/examples/svm_cmatrix.png)  

* Keeshond (the best class in F-score)  
![Keeshond](/examples/keeshond_3.jpg) 

* Staffordshire bull terrier (the worst class)  
![Staffordshire bull terrier](/examples/staffordshire_bull_terrier_13.jpg) 

* American pit bull terrier (sometimes predicted as staffordshire bull terrier)  
![American pit bull terrier](/examples/american_pit_bull_terrier_44.jpg) 
