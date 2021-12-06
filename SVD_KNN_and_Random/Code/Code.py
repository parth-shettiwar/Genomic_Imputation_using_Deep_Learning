import numpy as np
import matplotlib.pyplot as plt
import numpy as np
from sklearn.impute import KNNImputer
from sklearn.metrics import r2_score



def low_rank_approx(SVD, A, r):
    
    SVD = np.linalg.svd(A, full_matrices=True)
    u, s, v = SVD
    Ar = np.zeros((len(u), len(v)))
   
    new_u = u[:,:r]
    new_s = np.diag(s)[:r,:r]
    new_v = v[:,:r]
   
    Ar = np.matmul(np.matmul(new_u,new_s),new_v.T)

    return Ar

def post_process(A):
    A  = np.rint(A)
    A[np.where(A<0)] = 0
    A[np.where(A>2)] = 2
    return A

def SVD(mask_index,data_path):
    R = []

    acc_list = []

    n_ref = 200
    n_test_index = 400
    n_population = 500
    n_snips = 10000
    n_test_samples = n_population - n_test_index
    n_iters = 10
    rank = 10
    # mask_index = 100
    snip_chunk_size = 500
    Total_mask_snips = int((snip_chunk_size - mask_index)*(n_snips/snip_chunk_size))
    seed = 8

    np.random.seed(seed)

    data = np.genfromtxt(data_path, delimiter='\t')
    data = data[1:,:]
    data = data[:500,:]
    # print(data.shape)
    # print(np.sum(data==1))
    # print(np.sum(data==0))
    # print(np.sum(data==2))
    reference = data[:n_ref,:]
    test = data[n_test_index:,:]
    test_knn = test.copy()
    test_random = test.copy()
    miss_matrix = np.zeros(test.shape)
    gt_test = test.copy()

    for i in range(n_snips):
        if(i%n_population>=mask_index):
            test_random[:,i] = np.random.randint(3,size = (n_test_samples))
            test[:,i] = np.random.randint(3,size = (n_test_samples))
            test_knn[:,i] = np.nan
            miss_matrix[:,i] = 1


    test_final = np.concatenate((reference,test),axis = 0)
    test_final_knn = np.concatenate((reference,test_knn),axis = 0)
    test_random_final = np.concatenate((reference,test_random),axis = 0)
    # test_final_knn = test_final.copy()
    gt_test_final = np.concatenate((reference,gt_test),axis = 0)

    miss_final = np.zeros((n_ref + n_test_samples,n_snips))
    miss_final[n_ref:,:] = miss_matrix

    missing_array = np.where(miss_final==1)

    # print("Using Randomization")
    # print(np.sum(test_random_final[missing_array]==gt_test_final[missing_array]))
    # print(test_random_final[missing_array])
    acc_list.append((np.sum(test_random_final[missing_array]==gt_test_final[missing_array]))/Total_mask_snips)
    r2 = r2_score(gt_test_final[missing_array], test_random_final[missing_array],multioutput='variance_weighted')
    R.append(r2)
    # print("\n")
    for i in range(n_iters):
        temp_final = low_rank_approx(None, test_final,rank)   
        test_final[missing_array] = temp_final[missing_array]
        # print("Iteration = ",i)

    test_final = post_process(test_final)

    # print("Runnning Low Rank SVD Matrix Completion")
    # print(np.sum(test_final[missing_array]==gt_test_final[missing_array]))
    # print(test_final[missing_array].shape)
    acc_list.append(np.sum(test_final[missing_array]==gt_test_final[missing_array])/Total_mask_snips)
    r2 = r2_score(gt_test_final[missing_array], test_final[missing_array],multioutput='variance_weighted')
    R.append(r2)

    
    # print("Accuracy = ",np.sum(test_final[missing_array]==gt_test_final[missing_array])/Total_mask_snips)



    imputer = KNNImputer(n_neighbors=1,weights = "distance")
    common = imputer.fit_transform(test_final_knn)
    # print(result.shape)
    # print("Accuracy = ",
    # print(np.sum(common==test_final_knn))
    # print(np.sum(common[missing_array]==gt_test_final[missing_array]))
    
    acc_list.append(np.sum(common[missing_array]==gt_test_final[missing_array])/Total_mask_snips)
    r2 = r2_score(gt_test_final[missing_array],common[missing_array],multioutput='variance_weighted')
    R.append(r2)
    # print("acrrs",acc_list)

    return acc_list,R
# dsfwdf

# from sklearn.metrics import confusion_matrix
# ss = test_final[missing_array]
# test_y_true  = gt_test_final[missing_array]
# test_y_true = np.array(list(itertools.chain.from_iterable(test_y_true)))
# print(ss.shape)
# print(test_y_true.shape)  
# mat = confusion_matrix(test_y_true, ss, labels=[0,1,2])

# labels=['0','1','2']
# ax.matshow(mat, cmap=plt.cm.Blues)
# lis = [-1,0,1,2]
# ax.set_xticks(lis)
# ax.set_yticks(lis)
# ax.set_xticklabels([''] + labels)
# ax.set_yticklabels([''] + labels)
# for i in range(3):
#   for j in range(3):
#     c = mat[j,i]
#     ax.text(i, j, str(c), va='bottom', ha='center')
#     ax.text(i,j,str(round((((c)/(np.sum(mat,axis = 1)[j]))*100),2))+"%",va='top', ha='center')


# plt.xlabel('Predicted')
# plt.ylabel('True')
# plt.savefig('SVD.png')

print("Running on 10% MAF Dataset")
ind = [100]

for i in range(len(ind)):
    ACC,R = SVD(ind[i],"../Data/10MAF_data.txt")

print("R square Value of Randomization, SVD, KNN = ", R)
print("Accuracy of Randomization, SVD, KNN = ", ACC)

print("\n")

    


lis4 = []
lis5 = []
lis6 = []

print("Running on 60% MAF Dataset")
for i in range(len(ind)):
    ACC,R = SVD(ind[i],"../Data/60MAF_data.txt")

print("R square Value of Randomization, SVD, KNN = ", R)
print("Accuracy of Randomization, SVD, KNN = ", ACC)

    




