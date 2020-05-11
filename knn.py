import numpy as np
import operator as op
import matplotlib.pyplot as plt
from minst import *

def autonorm(dataset,flag):#if flag =1 return binumerical dataset else return normal norm dataset
    norm_dataset=dataset.copy()
    minvals=np.min(dataset,axis=0)
    maxvals=np.max(dataset,axis=0)
    diffvals=maxvals-minvals
    for i in range(diffvals.shape[0]):
        for j in range(diffvals.shape[1]):
            if diffvals[i,j]==0:
                diffvals[i,j]=1
    n=dataset.shape[0]
    for i in range(n):
        norm_dataset[i]=(norm_dataset[i]-minvals)/diffvals
    if flag==0:
        return norm_dataset
    else:
        return np.where(norm_dataset>=0.5,1,0)

def class_handwrite_by_knn(k,flag=0):
    """main function"""
    #load training data
    train_images = load_train_images()
    train_labels = load_train_labels()
    test_images = load_test_images()
    test_labels = load_test_labels()
    #knn begin
    #normalize the matrix
    norm_train_images=autonorm(train_images,flag)
    norm_test_images=autonorm(test_images,flag)
    #flatten the normalize one
    train_images_dataset=flatten_set(norm_train_images)
    test_images_dataset=flatten_set(norm_test_images)
    #print(train_images_dataset)
    #answer labels
    answer_labels=np.empty(test_labels.shape,dtype=type(test_labels[0]))
    #get answer by knn
    for i in range(10):
        answer_labels[i]=knn_classifier(test_images_dataset[i],train_images_dataset,train_labels,k)
        print("knn give:"+str(answer_labels[i])+"true labels"+str(test_labels[i]))
    #print(accuracy_rate_cal(answer_labels,test_labels))


def flatten_set(dataset):
    flatten_dataset=np.empty((dataset.shape[0],dataset.shape[1]*dataset.shape[2]))
    for i in range(dataset.shape[0]):
        flatten_dataset[i]=dataset[i].flatten()
    return flatten_dataset

def knn_classifier(exp,dataset,labels,k):
    """return the type of the exp base on dataset and labels"""
    datasize=dataset.shape[0]
    class_count={}
    if k>datasize:
        return 0
    cal_mat=np.tile(exp,(datasize,1))
    diff_mat=cal_mat-dataset
    sq_diff_mat=diff_mat**2
    sq_distance=sq_diff_mat.sum(axis=1)
    distance=sq_distance**0.5
    sorted_distanceindices=distance.argsort()
    for i in range(10,20):
        current_label=labels[sorted_distanceindices[i]]
        class_count[current_label]=class_count.get(current_label,0)+1
    rec_list=sorted(class_count.items(),key=op.itemgetter(1),reverse=True)
    return rec_list[0][0]




def accuracy_rate_cal(label_true,label_get):
    """return the accuracy rate in knn"""
    diff_label=label_true-label_get
    true_list=np.where(diff_label==0,1,0)
    return true_list.sum()/diff_label.shape[0]


class_handwrite_by_knn(3,1)

