import numpy as np
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


n_ref = 200
n_test_index = 400
n_population = 500
n_snips = 10000
n_test_samples = n_population - n_test_index
n_iters = 10
rank = 10
mask_index = 100
snip_chunk_size = 500
Total_mask_snips = int((snip_chunk_size - mask_index)*(n_snips/snip_chunk_size))
seed = 1

np.random.seed(seed)

data = np.genfromtxt("../Data/mat.txt", delimiter='\t')
data = data[1:,:]
reference = data[:n_ref,:]
test = data[n_test_index:,:]
miss_matrix = np.zeros(test.shape)
gt_test = test.copy()

for i in range(n_snips):
    if(i%n_population>=mask_index):
        test[:,i] = np.random.randint(3,size = n_test_samples)
        miss_matrix[:,i] = 1


test_final = np.concatenate((reference,test),axis = 0)
gt_test_final = np.concatenate((reference,gt_test),axis = 0)

miss_final = np.zeros((n_ref + n_test_samples,n_snips))
miss_final[n_ref:,:] = miss_matrix

missing_array = np.where(miss_final==1)

print("Using Randomization")
print("Accuracy = ", (np.sum(test_final[missing_array]==gt_test_final[missing_array]))/Total_mask_snips)
print("\n")
for i in range(n_iters):
    temp_final = low_rank_approx(None, test_final,rank)   
    test_final[missing_array] = temp_final[missing_array]
    print("Iteration = ",i)
    
test_final = post_process(test_final)

print("Using Low Rank SVD Matrix Completion")
print(np.sum(test_final[missing_array]==gt_test_final[missing_array]))
print("Accuracy = ",np.sum(test_final[missing_array]==gt_test_final[missing_array])/Total_mask_snips)