#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar 18 15:30:42 2018

@authors: Nezar Ahmed Mukhtar
          Dina Abdel-raouuf
          Rania Mahmoud Hassan
          Noha Ahmed Darwish
          Rana Shaker
                   
"""
"""
                                          *************                                      
                                         *             *
======================================>  * [ READ ME ] * <========================================== 
                                         *             *
                                          *************   
For Part 1 and Part 2:
    1-User input message will appear asking for either using PCA OR DCT.
    2-To enter total number of clusters, a user input message will appear asking for number of clusters needed.
    3-Part 2 is related to Part 1 so either to comment both or uncomment them.
    
For Part 3:
    1-Part 3 is on its own so you can comment both Part 1 and Part 2 when running Part 3.
    
                            ##########################                    ###########################
Each Part begins with ==>   ### Part no.: function ###  and ends with ==> ##### End of Part no. #####
                            ##########################                    ###########################    
"""
import numpy as np
import scipy.io
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import confusion_matrix
from sklearn.mixture import GaussianMixture
import time
from scipy.fftpack import dct as dct
import zigzag as zig
from sklearn import svm, metrics


##################################DCT 2D#################################
def dct2d(x,classes,images_per_class):
    t = 2 
    DCT_c = []
    for i in range(classes * images_per_class):
            temp = dct(x[i],type=t,norm='ortho').transpose()
            D_i=dct(temp,type=t,norm='ortho').transpose()
            DCT_c.append(zig.zigzag(D_i)[0:20])
    return DCT_c
###################################End of DCT 2D #######################


############################################################################################
################################## Preprocessing of data ###################################
############################################################################################

train_and_test_data = scipy.io.loadmat('ReducedMNIST.mat')
train_data=train_and_test_data["SmallTrainData"]
test_data=train_and_test_data["SmallTestData"]

number_of_classes=10
train_images=[]
train_classes=[]
test_images=[]
test_classes=[]

train_images_norm=[]
test_images_norm=[]
train_classes_appended=[]
test_classes_appended=[]
TrainImg=[]
TestImg=[]
TrainLabels=[]
TestLabels=[]
TrainLabels_h=[]
TestLabels_h=[]
reduced_TrainData_rpca=[]
reduced_TestData_rpca=[]
x=0

#train_images_norm=np.asarray(train_images_norm)
"""
     1-Segregation of Images from classes  
     2-Standardize the dataset’s features onto unit scale (mean = 0 and variance = 1) which is 
     a requirement for the optimal performance of many machine learning algorithms.
"""

for i in range (number_of_classes):
     train_images.append(train_data[i,0])
     train_classes.append(train_data[i,1])
     test_images.append(test_data[i,0])
     test_classes.append(test_data[i,1])

train_images=(np.asarray(train_images))
test_images=(np.asarray(test_images))
train_classes=np.asarray(train_classes)
test_classes=np.asarray(test_classes)     

for i in range (number_of_classes): 
     for j in range (1000):
         train_images_temp = StandardScaler().fit_transform(train_images[i,j]) 
         train_images_norm.append(train_images_temp)
         train_classes_appended.append(train_classes[i,j])
     for k in range (100):
         test_images_temp = StandardScaler().fit_transform(test_images[i,k])    
         test_images_norm.append(test_images_temp)
         test_classes_appended.append(test_classes[i,k])


train_images_norm_flatten=np.reshape(np.asarray(train_images_norm),(10000,784))
test_images_norm_flatten=np.reshape(np.asarray(test_images_norm),(1000,784))
         
#### hot-one decoding ####
test_classes_true=[np.where(r==1)[0][0] for r in test_classes_appended]
train_classes_true=[np.where(r==1)[0][0] for r in train_classes_appended]

################################################################################################
################################### End of Preprocessing #######################################
################################################################################################







################################################################################################
################################### PART1: Features ############################################
################################################################################################

################################################################################################
############################################ DCT ###############################################
################################################################################################
DCT_coeffs = []
DCT_coeffs_0 = []
DCT_size = 20
DCT_coeffs=dct2d(train_images_norm,10,1000)
DCT_coeffs_0=dct2d(test_images_norm,10,100)
################################################################################################
######################################## End of DCT ############################################
################################################################################################



################################################################################################
########################################### PCA ################################################
################################################################################################

print("\n[PART1]\nVariance Check:(Check that all variances exceeds 90%)")

n_features=156
pca=PCA(n_components=n_features)
pca.fit(train_images_norm_flatten)

##### Apply PCA to train_set ######

train_pca=pca.transform(train_images_norm_flatten)
var_train_check=pca.explained_variance_ratio_.cumsum()

##### Apply PCA to test_set ######
test_pca=pca.transform(test_images_norm_flatten)
var_test_check=pca.explained_variance_ratio_.cumsum()

#### printing results ####
print("\nTrain Variance= ",var_train_check[-1]*100," %")
print("Test Variance=  ",var_test_check[-1]*100," %")
print("\nNumber of features after reduction using PCA is ",n_features)


################################################################################################
####################################### End of PCA #############################################
################################################################################################
  
###################################User input#######################################

IN = input("Enter PCA or DCT ")
while(1):        
        if IN == 'DCT':
           train_coeff = DCT_coeffs
           test_coeff = DCT_coeffs_0
           break
        elif IN == 'PCA':
             train_coeff = train_pca
             test_coeff = test_pca
             break
        else:
             IN = input('Enter PCA or DCT only and upper case')    

#####################################End of user input######################       

###############################################################################################
###################################### End of PART1 ###########################################
###############################################################################################








###############################################################################################
################################## PART2: Classifying #########################################
###############################################################################################

###############################################################################################
################################## Kmeans clustering $#########################################
###############################################################################################
print("\n\n[PART2] with [Part6]\n")


print("--------------------[K_MEANS CLUSTERING]-------------------------")

"""
PLEASE ENTER TOTAL NUMBER OF CLUSTERS IN VARIABLE ==> n_clusters
"""
n_clusters=int(input("Enter total number of clusters needed:"))
start_time_clustering = time.time()


#### K-means clustering ###


if(n_clusters%2 != 0) : raise ValueError('Enter an even integer non-zero number')
n_clusters_per_class=int(n_clusters/10)
print("Number of clusters per class is ",n_clusters_per_class," clusters")
y_labels_true=[]

kmeans = KMeans(n_clusters=n_clusters,n_init=10,max_iter=5000,algorithm='full',random_state=0)
y_labels_train=kmeans.fit_predict(train_coeff)
centeroids=kmeans.cluster_centers_
y_labels_train=np.asarray(y_labels_train)

clusters_used=[]
for i in range (10):
    counts = np.bincount(y_labels_train[i*1000:(i*1000)+1000-1])
    for j in range (100):
         clusters=np.argpartition(counts, -1*n_clusters_per_class)[-1*n_clusters_per_class:]
         y_labels_true.append(clusters)
    clusters_used.append(clusters)    

#### Predicting on test set ####
test_classes_pred=kmeans.predict(test_coeff)


##### Accuracy calculation ####
error=0
for i in range (1000):
    if(test_classes_pred[i] not in y_labels_true[i] ) : error=error+1

acc=1-(error/1000)    

#### confusion matrix ####
if(n_clusters_per_class==1):
    conf_mat=confusion_matrix(y_labels_true,test_classes_pred)
    conf_mat=np.asarray(conf_mat)
    print("\n Confusion Matrix")
    print(conf_mat)
#### printing results ####

print("\nAccuracy of prediction= ",acc*100,"%")
print("Processing time=%s seconds\n\n" % (time.time() - start_time_clustering))


###############################################################################################
################################## End of Kmeans clustering ###################################
###############################################################################################




###############################################################################################
################################## Gaussian Mixture Model #####################################
###############################################################################################

print("\n--------------------[GAUSSIAN MIXTURE MODEL]-----------------------")

"""
PLEASE ENTER TOTAL NUMBER OF GMMs IN VARIABLE ==> n_GMM
"""
n_GMM=int(input("Enter total number of clusters needed:"))
start_time_GMM = time.time()

##### GMM ####

if(n_GMM%2 != 0) : raise ValueError('Enter an even integer non-zero number')
n_GMM_per_class=int(n_GMM/10)
print("Number of GMM per class is ",n_GMM_per_class," GMMs")


GMM=GaussianMixture(n_components=n_GMM,n_init=10,max_iter=5000,covariance_type='full',random_state=0)
GMM.fit(train_coeff)
weights=GMM.weights_
means=GMM.means_
summation_check=sum(weights)

probs_train=GMM.predict_proba(train_coeff)
#print(probs.round(3))
probs_test=GMM.predict_proba(test_coeff)
probs_test_maxes=np.argmax(probs_test,axis=1)


y_labels_true=[]
GMM_used=[]
probs_train_maxes=np.argmax(probs_train,axis=1)
for i in range (10):
    counts = np.bincount(probs_train_maxes[i*1000:(i*1000)+1000-1])
    for j in range (100):
         GMMs=np.argpartition(counts, -1*n_GMM_per_class)[-1*n_GMM_per_class:]        
         y_labels_true.append(GMMs)
    GMM_used.append(GMMs)    


##### Accuracy calculation ####
error=0

for i in range (1000):
    if(probs_test_maxes[i] not in y_labels_true[i] ) : error=error+1

acc=1-(error/1000)    

#### confusion matrix ####
if(n_GMM_per_class==1):
    conf_mat=confusion_matrix(y_labels_true,probs_test_maxes)
    conf_mat=np.asarray(conf_mat)
    print("\n Confusion Matrix")
    print(conf_mat)

#### printing results ####

print("\nAccuracy of prediction= ",acc*100,"%")
print("Processing time=%s seconds" % (time.time() - start_time_GMM))
#
#
##################################################################################################
################################ End of Gaussian Mixture Model ##################################
#################################################################################################
#
#
#
#################################################################################################
######################################## End of PART2 ###########################################
#################################################################################################




############# show images ################ 
#for i in range (0,1000,25):
#  print("loop",i) 
#  plt.imshow(test_images_norm[i],cmap='gray')
#  plt.title(test_classes_pred[i])
#  plt.show()
#  input()

################################################################################################
################################### PART3(a): Concatenating Features ############################
################################################################################################

for i in range(10):
    TrainImg.append(train_data[i][0])
    TestImg.append(test_data[i][0])
    TrainLabels.append(train_data[i][1])
    TestLabels.append(test_data[i][1])



#1
train_images_norm=[]
train_classes_appended=[]
test_images_norm=[]
test_classes_appended=[]

#2
DCT_coeffs=[] 
DCT_coeffs_0=[]
DCT_size=100

train_images=(np.asarray(TrainImg))
test_images=(np.asarray(TestImg))
train_classes=np.asarray(TrainLabels)
test_classes=np.asarray(TestLabels)     
for i in range (10): 
     for j in range (1000):
         train_images_temp = StandardScaler().fit_transform(train_images[i,j]) 
         train_images_norm.append(train_images_temp)
         train_classes_appended.append(train_classes[i,j])
         x=dct(dct(train_images_temp.T).T)
         x_zig=zig.zigzag(x)   
         DCT_coeffs.append(x_zig[0:DCT_size])
          
          
         
     for k in range (100):
         test_images_temp = StandardScaler().fit_transform(test_images[i,k])    
         test_images_norm.append(test_images_temp)
         test_classes_appended.append(test_images[i,k])
         x_0=dct(dct(test_images_temp.T).T)
         x0_zig=zig.zigzag(x_0)   
         DCT_coeffs_0.append(x0_zig[0:DCT_size])
         
train_images_norm_flatten=np.reshape(np.asarray(train_images_norm),(10000,784))
test_images_norm_flatten=np.reshape(np.asarray(test_images_norm),(1000,784))

TrainLabels=np.reshape(TrainLabels,(10000,10))
TestLabels=np.reshape(TestLabels,(1000,10))

for i in range(len(TrainLabels)):
   x=np.argmax(TrainLabels[i])
   TrainLabels_h.append(x)

for i in range(len(TestLabels)):
   x=np.argmax(TestLabels[i])
   TestLabels_h.append(x)

##### Apply PCA to train_set ######
n_features=100
pca1=PCA(n_components=n_features)
train_pca=pca1.fit_transform(train_images_norm_flatten)
var_train_check=pca1.explained_variance_ratio_.cumsum()
print("-------------For part 3 (a)/ Concatenating Features----------")
print("\n\nTrain Variance= ",var_train_check[-1]*100," %")

##### Apply PCA to test_set ######
test_pca=pca1.transform(test_images_norm_flatten)
var_test_check=pca1.explained_variance_ratio_.cumsum()
print("Test Variance=  ",var_test_check[-1]*100," %")
   
##### concatenating features ######
x_new_train=[DCT_coeffs,train_pca]     ## for train data set
x_new_test=[DCT_coeffs_0,test_pca]     ##for test data set
weightMatrix = []
accuracyAppendedSVM = []
accuracyAppendedKMeans = []
accuracyAppendedGMM = []
features_train=[]
features_test=[]
conf_matrices1=[]
process_time_mat1=[]
process_time_mat2=[]
conf_matrices2=[]
conf_matrices3=[]
process_time_mat3=[]
best_features_train=[]
best_features_test=[]
#### Concatenation Part ####
for i in range (1,9):
    weight=np.divide(i,10)
    weightMatrix.append(weight)
    
    l1=(np.array(x_new_train[0], dtype=float)*weight)
    l2=(np.array(x_new_train[1], dtype=float)*(1-weight))
    l3=(np.array(x_new_test[0], dtype=float)*weight)
    l4=(np.array(x_new_test[1], dtype=float)*(1-weight))
    weighted_x_train=np.hstack((l1,l2))
    weighted_x_test=np.hstack((l3,l4))
    
    features_train.append(weighted_x_train)
    features_test.append(weighted_x_test)
    
    #### testing with SVM ############
    start_time_processing=time.time()
    classifier_Non = svm.SVC(kernel="poly", C=5,cache_size=800,max_iter=10000 )
    classifier_Non.fit(weighted_x_train, TrainLabels_h)
    expected =TestLabels_h 
    predicted = classifier_Non.predict(weighted_x_test)
    accuracy_SVM = metrics.accuracy_score(expected, predicted)
    accuracyAppendedSVM.append(accuracy_SVM)
    #### confusion matrix List for SVM #####
    conf_mat1= metrics.confusion_matrix(expected, predicted)
    conf_matrices1.append(conf_mat1)
    #### Processing time list for SVM ####
    process_time1=time.time() - start_time_processing
    process_time_mat1.append(process_time1)
    
    ###### testing with GMM #########
    start_time_GMM = time.time()
    n_GMM=10
    if(n_GMM%2 != 0) : raise ValueError('Enter an even integer non-zero number')
    n_GMM_per_class=int(n_GMM/10)
    GMM=GaussianMixture(n_components=n_GMM,n_init=10,max_iter=5000,covariance_type='full',random_state=0)
    GMM.fit(weighted_x_train)
    weights=GMM.weights_
    means=GMM.means_
    summation_check=sum(weights)
    
    probs_train=GMM.predict_proba(weighted_x_train)
    probs_test=GMM.predict_proba(weighted_x_test)
    probs_test_maxes=np.argmax(probs_test,axis=1)
    
    
    y_labels_true=[]
    GMM_used=[]
    probs_train_maxes=np.argmax(probs_train,axis=1)
    for i in range (10):
        counts = np.bincount(probs_train_maxes[i*1000:(i*1000)+1000-1])
        for j in range (100):
             GMMs=np.argpartition(counts, -1*n_GMM_per_class)[-1*n_GMM_per_class:]        
             y_labels_true.append(GMMs)
        GMM_used.append(GMMs)    
    ##### Accuracy calculation ####    
    error1=0

    for i in range (1000):
        if(probs_test_maxes[i] not in y_labels_true[i] ) : error1=error1+1

    acc1=1-(error1/1000)
    accuracyAppendedGMM.append(acc1)
     ####3 confusion matrix list for GMM #####
    if(n_GMM_per_class==1):
        conf_mat2=confusion_matrix(y_labels_true,probs_test_maxes)
        conf_mat2=np.asarray(conf_mat2)
        conf_matrices2.append(conf_mat2)
    #### Processing time list for GMM #####
    process_time2=time.time() - start_time_GMM
    process_time_mat2.append(process_time2)
        
    #### testing with KMeans ##########
    start_time_clustering = time.time()
    n_clusters=10
    if(n_clusters%2 != 0) : raise ValueError('Enter an even integer non-zero number')
    n_clusters_per_class=int(n_clusters/10)
    y_labels_true=[]
    kmeans = KMeans(n_clusters=n_clusters,n_init=10,max_iter=5000,algorithm='full',random_state=0)
    y_labels_train=kmeans.fit_predict(weighted_x_train)
    centeroids=kmeans.cluster_centers_
    y_labels_train=np.asarray(y_labels_train)
    
    clusters_used=[]
    for i in range (10):
        counts = np.bincount(y_labels_train[i*1000:(i*1000)+1000-1])
        for j in range (100):
             clusters=np.argpartition(counts, -1*n_clusters_per_class)[-1*n_clusters_per_class:]
             y_labels_true.append(clusters)
        clusters_used.append(clusters)    
    
    #### Predicting on test set ####
    test_classes_pred=kmeans.predict(weighted_x_test)
    
    
    ##### Accuracy calculation ####
    error2=0
    for i in range (1000):
        if(test_classes_pred[i] not in y_labels_true[i] ) : error2=error2+1
    
    acc2=1-(error2/1000)
    accuracyAppendedKMeans.append(acc2)
    
    ####3 confusion matrix list for KMeans #####
    if(n_clusters_per_class==1):
        conf_mat3=confusion_matrix(y_labels_true,test_classes_pred)
        conf_mat3=np.asarray(conf_mat3)
        conf_matrices3.append(conf_mat3)
    #### Processing time list for KMeans ####
    process_time3=time.time() - start_time_clustering
    process_time_mat3.append(process_time3)

#### Printing Results #####
##SVM
plt.figure()
plt.plot(weightMatrix,accuracyAppendedSVM,'ko')
plt.title("Accuracy vs. weights for SVM algorithm")
print("\n\nBest accuracy for SVM = %0.4f %c at weight = %0.1f " % ((accuracyAppendedSVM[np.argmax(accuracyAppendedSVM)]*100), '%' ,weightMatrix[np.argmax(accuracyAppendedSVM)]))
print("Processing time=%0.7f seconds\n\n" % (process_time_mat1[np.argmax(accuracyAppendedSVM)]))
print("\nConfusion Matrix for weight = %0.1f is" % weightMatrix[np.argmax(accuracyAppendedSVM)])
print(conf_matrices1[np.argmax(accuracyAppendedSVM)])    

##GMM
plt.figure()
plt.plot(weightMatrix,accuracyAppendedGMM,'ko')
plt.title("Accuracy vs. weights for GMM algorithm")
print("\n\nBest accuracy for GMM = %0.4f %c at weight = %0.1f " % ((accuracyAppendedGMM [np.argmax(accuracyAppendedGMM)]*100), '%' ,weightMatrix[np.argmax(accuracyAppendedGMM)]))
print("Processing time=%0.7f seconds\n\n" % (process_time_mat2[np.argmax(accuracyAppendedGMM)]))
if(n_GMM_per_class==1):    ##### confusion matrix ####
    print("\nConfusion Matrix for weight = %0.1f is" % weightMatrix[np.argmax(accuracyAppendedGMM)])
    print(conf_matrices2[np.argmax(accuracyAppendedGMM)])
print("Number of GMM per class is ",n_GMM_per_class," GMMs")
best_features_train[2]=features_train[np.argmax(accuracyAppendedGMM)]
best_features_test[2]=features_test[np.argmax(accuracyAppendedGMM)]

##KMeans
plt.figure()
plt.plot(weightMatrix,accuracyAppendedKMeans,'ko')
plt.title("Accuracy vs. weights for KMeans algorithm")
print("\n\nBest accuracy for KMeans = %0.4f %c at weight = %0.1f " % ((accuracyAppendedKMeans [np.argmax(accuracyAppendedKMeans)]*100), '%' ,weightMatrix[np.argmax(accuracyAppendedKMeans)]))
print("Processing time=%0.7f seconds\n\n" % (process_time_mat3[np.argmax(accuracyAppendedKMeans)]))
if(n_clusters_per_class==1):
    print("\nConfusion Matrix for weight = %0.1f is" % weightMatrix[np.argmax(accuracyAppendedKMeans)])
    print(conf_matrices3[np.argmax(accuracyAppendedKMeans)])

################################################################################################
####################################### End of PART3(a) ########################################
################################################################################################
    
################################################################################################
####################################### PART3 (b): Diagonalization #############################
################################################################################################
print("\n----------[Using PCA to diagonalize Covariance matrix of features]-------------")
best_features_train=features_train[np.argmax(accuracyAppendedSVM)]
best_features_test=features_test[np.argmax(accuracyAppendedSVM)]
n_features2=70
pca=PCA(n_components=n_features2)
pca.fit(best_features_train)

##### Apply PCA to train_set ######
weighted_x_train_pca=pca.transform(best_features_train)
weighted_x_train_var=pca.explained_variance_ratio_.cumsum()

#### Apply PCA to test_set ######
weighted_x_test_pca=pca.transform(best_features_test)
weighted_x_test_var=pca.explained_variance_ratio_.cumsum()

#### printing results ####
print("\nTrain Variance= ",weighted_x_train_var[-1]*100," %")
print("Test Variance=  ",weighted_x_test_var[-1]*100," %")
print("\nNumber of features after reduction using PCA is ",n_features2)
div=np.diag(np.linalg.eigvals(np.cov(weighted_x_train_pca.T)))
print("\nCovariance matrix diagonalized",div)
    
########### testing with SVM ############
best_features_train=features_train[np.argmax(accuracyAppendedSVM)]
best_features_test=features_test[np.argmax(accuracyAppendedSVM)]
pcaSVM=PCA(n_components=n_features2)
pcaSVM.fit(best_features_train)

##### Apply PCA to train_set ######
weighted_x_train_pca=pcaSVM.transform(best_features_train)
weighted_x_train_var=pcaSVM.explained_variance_ratio_.cumsum()

#### Apply PCA to test_set ######
weighted_x_test_pca=pcaSVM.transform(best_features_test)
weighted_x_test_var=pca.explained_variance_ratio_.cumsum()
print("Training and Testing for SVM\n")
start_time_processing_D=time.time()
classifier_Non_D = svm.SVC(kernel="poly", C=5,cache_size=800,max_iter=10000 )
classifier_Non_D.fit(weighted_x_train_pca, TrainLabels_h)
expected_D =TestLabels_h 
predicted_D = classifier_Non_D.predict(weighted_x_test_pca)
accuracy_SVM_D = metrics.accuracy_score(expected_D, predicted_D)
percentage=np.multiply(accuracy_SVM_D,100)
print("\nAccuracy: %0.4f" % percentage)
print("Processing time=%0.7f seconds\n\n" % (time.time() - start_time_processing_D))
print("Confusion matrix:\n%s" % metrics.confusion_matrix(expected_D ,predicted_D))

#
############# testing with GMM ##############
print("Training and Testing for GMM\n")
best_features_train=features_train[np.argmax(accuracyAppendedGMM)]
best_features_test=features_test[np.argmax(accuracyAppendedGMM)]
pcaGMM=PCA(n_components=n_features2)
pcaGMM.fit(best_features_train)
weighted_x_train_pca2=pca.transform(best_features_train)
weighted_x_train_var2=pca.explained_variance_ratio_.cumsum()

#### Apply PCA to test_set ######
weighted_x_test_pca2=pca.transform(best_features_test)
weighted_x_test_var2=pca.explained_variance_ratio_.cumsum()
start_time_GMM2_D = time.time()
n_GMM2=40
if(n_GMM2%2 != 0) : raise ValueError('Enter an even integer non-zero number')
n_GMM_per_class2=int(n_GMM2/10)
GMM2=GaussianMixture(n_components=n_GMM,n_init=10,max_iter=5000,covariance_type='full',random_state=0)
GMM2.fit(weighted_x_train_pca2)
weights_D=GMM.weights_
means_D=GMM.means_
summation_check=sum(weights_D)
    
probs_train_D=GMM2.predict_proba(weighted_x_train_pca2)
probs_test_D=GMM2.predict_proba(weighted_x_test_pca2)
probs_test_maxes_D=np.argmax(probs_test_D,axis=1)
    
    
y_labels_true_D=[]
GMM_used_D=[]
probs_train_maxes_D=np.argmax(probs_train_D,axis=1)
for i in range (10):
    counts_DGMM = np.bincount(probs_train_maxes_D[i*1000:(i*1000)+1000-1])
    for j in range (100):
        GMMs=np.argpartition(counts_DGMM, -1*n_GMM_per_class2)[-1*n_GMM_per_class2:]        
        y_labels_true_D.append(GMMs)
    GMM_used_D.append(GMMs)
        
    ##### Accuracy calculation ####    
error1_D=0

for i in range (1000):
    if(probs_test_maxes_D[i] not in y_labels_true_D[i] ) : error1_D=error1_D+1

acc1_D=1-(error1_D/1000)
    
##### printing results ####

print("\nAccuracy of prediction= ",acc1_D*100,"%")
print("Processing time=%s seconds" % (time.time() - start_time_GMM2_D))
if(n_GMM_per_class2==1):    ##### confusion matrix ####
    conf_mat2_D=confusion_matrix(y_labels_true_D,probs_test_maxes_D)
    conf_mat2=np.asarray(conf_mat2_D)
    print("\n Confusion Matrix")
    print(conf_mat2_D)

#
######## testing with KMeans ##########
print("Training and Testing for KMeans\n")
best_features_train=features_train[np.argmax(accuracyAppendedKMeans)]
best_features_test=features_test[np.argmax(accuracyAppendedKMeans)]
pcaK=PCA(n_components=n_features2)
pcaK.fit(best_features_train)

##### Apply PCA to train_set ######
weighted_x_train_pca3=pcaK.transform(best_features_train)
weighted_x_train_var3=pcaK.explained_variance_ratio_.cumsum()

#### Apply PCA to test_set ######
weighted_x_test_pca3=pcaK.transform(best_features_test)
weighted_x_test_var3=pcaK.explained_variance_ratio_.cumsum()
start_time_processing3_D=time.time()
n_clusters2=10
if(n_clusters2%2 != 0) : raise ValueError('Enter an even integer non-zero number')
n_clusters_per_class2=int(n_clusters2/10)
y_labels_true2=[]
kmeans2 = KMeans(n_clusters=n_clusters2,n_init=10,max_iter=5000,algorithm='full',random_state=0)
y_labels_train2=kmeans2.fit_predict(weighted_x_train_pca3)
centeroids2=kmeans2.cluster_centers_
y_labels_train2=np.asarray(y_labels_train2)
            
clusters_used2=[]
for i in range (10):
   counts2 = np.bincount(y_labels_train2[i*1000:(i*1000)+1000-1])
   for j in range (100):
      clusters2=np.argpartition(counts2, -1*n_clusters_per_class2)[-1*n_clusters_per_class2:]
      y_labels_true2.append(clusters2)
      clusters_used2.append(clusters2)    
            
            #### Predicting on test set ####
test_classes_pred2=kmeans2.predict(weighted_x_test_pca3)
            
            
            ##### Accuracy calculation ####
error2_D=0
for i in range (1000):
   if(test_classes_pred2[i] not in y_labels_true2[i] ) : error2_D=error2_D+1
            
acc2_D=1-(error2_D/1000)

#### printing results ####

print("\nAccuracy of prediction= ",acc2_D*100,"%")
print("Processing time=%s seconds\n\n" % (time.time() - start_time_processing3_D))
if(n_clusters_per_class2==1):    #### confusion matrix ####
    conf_mat3_D=confusion_matrix(y_labels_true2,test_classes_pred2)
    conf_mat3_D=np.asarray(conf_mat3_D)
    print("\n Confusion Matrix")
    print(conf_mat3_D)
#
#################################################################################################
######################################## End of PART3(b) ########################################
#################################################################################################   
