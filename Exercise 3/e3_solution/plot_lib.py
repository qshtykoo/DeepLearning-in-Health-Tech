"""
Copyright 2017 (c) Zheng Zhao, All rigths reserved
Department of Electrial Engineering and Automation
Aalto University
zheng.zhao@aalto.fi
zz@zabemon.com
"""
import numpy as np
import itertools
import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn.metrics import confusion_matrix, precision_score, f1_score
from sklearn.preprocessing import label_binarize
from sklearn.metrics import precision_recall_curve, average_precision_score
from scipy import interp
from itertools import cycle

def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    https://www.kaggle.com/grfiv4/plot-a-confusion-matrix
    """
    if normalize:
        cm2 = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')
        cm2 = cm

    print(cm)
    plt.imshow(cm2, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes)
    plt.yticks(tick_marks, classes)

    fmt = '.1f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], 'd')  + '\n(' + format(cm2[i, j]*100, fmt) + ')',
                 horizontalalignment="center", verticalalignment='center',
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.show()

def get_mean_prc(y_true, y_pred):
    '''
    The function will return precision, recall curve and auc for all classes
    http://scikit-learn.org/stable/auto_examples/model_selection/plot_precision_recall.html
    '''
    n_class = y_true.shape[-1]
    precision = dict()
    recall = dict()
    average_precision = dict()

    # y_true = np.concatenate(y_true)
   #  y_pred = np.concatenate(y_pred)
    for i in range(n_class):
        precision[i], recall[i], _ = precision_recall_curve(y_true[:, i], y_pred[:, i])
        average_precision[i] = average_precision_score(y_true[:, i], y_pred[:, i])

    precision["micro"], recall["micro"], _ = precision_recall_curve(y_true.ravel(), y_pred.ravel())
    average_precision["micro"] = average_precision_score(y_true, y_pred, average="micro")
    print('Average precision score, micro-averaged over all classes: {0:0.2f}'.format(average_precision["micro"]))

    return precision, recall, average_precision