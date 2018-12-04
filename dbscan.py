
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from collections import defaultdict


# In[2]:


lines = []

with open("data/train.dat", "r") as fh:
    lines = fh.readlines()


# In[45]:


print(len(lines))


# In[4]:


val = []
ind = []
ptr = [0]
inddict = dict()


# In[5]:


index = 0
for j in range(0, len(lines)):
    d = lines[j].split(" ")  
    for i in range(0, len(d), 2):        
        if not inddict.has_key(d[i]):
            inddict[d[i]]= index
            index+=1
        ind.append(inddict.get(d[i]))
        val.append(d[i+1])
    ptr.append(ptr[len(ptr)-1]+len(d)/2)
            


# In[6]:


print(len(val))
print(len(ind))
print(len(ptr))


# In[7]:


print(val[:5])
print(ind[:5])
print(ptr[:5])
print(len(inddict))
# print(len(set(ind)))


# In[8]:


from scipy.sparse import csr_matrix


# In[9]:


ncols = len(set(ind))
nrows = len(lines)
# mat = csr_matrix((val, ind, ptr), shape=(nrows, ncols), dtype=np.long)
mat = csr_matrix((val, ind, ptr), dtype=np.long)


# In[10]:


print(mat.shape)
print("mat:", mat[:5,:20].todense(), "\n")


# In[11]:


from sklearn.decomposition import TruncatedSVD


# In[12]:


svd = TruncatedSVD(n_components=30, n_iter=7, random_state=42)
reduced_mat = svd.fit_transform(mat)


# In[13]:


variance = svd.explained_variance_
ratio = svd.explained_variance_ratio_
print(ratio)
print(variance)

import matplotlib.pyplot as plt
plt.rcParams["figure.figsize"] = [16,9]
plt.plot(variance)
plt.show()


# In[14]:


print(type(reduced_mat))
print(reduced_mat.shape)
print(reduced_mat[:2])


# In[15]:


svd = TruncatedSVD(n_components=8, n_iter=7, random_state=42)
reduced_mat = svd.fit_transform(mat)


# In[16]:


print(type(reduced_mat))
print(reduced_mat.shape)
print(reduced_mat[:5])


# In[17]:


from scipy.spatial.distance import pdist, squareform


# In[18]:


dist_mat1 = pdist(reduced_mat, metric='euclidean')
dist_mat = squareform(dist_mat1)


# In[19]:


print(dist_mat.shape)


# In[76]:


dist_mat[:5,:5]


# In[21]:


# def findkNeibour(k):
#     list = []
#     for row in dist_mat:
#         row.sort()
# #         list.append(row[1:k+1])
#         list.append(row[1:])
#     return list
# #     print(row.tolist().sort())


# In[22]:


# list_k1 = findkNeibour(2)


# In[23]:


# import matplotlib.pyplot as plt
# plt.plot(list_k1)
# plt.show()


# In[24]:


from sklearn.preprocessing import normalize


# In[25]:


# norm_dist_mat= normalize(dist_mat, norm='l2')


# In[26]:


# print(norm_dist_mat.shape)


# In[27]:


dist_mat_sorted = dist_mat.copy()
# norm_dist_mat_sorted = norm_dist_mat.copy()


# In[28]:


for row in dist_mat_sorted:
    row.sort()


# In[29]:


print(dist_mat_sorted.shape)


# In[30]:


# def findkNeibour(k):
#     list = []
#     for row in norm_dist_mat_copy:
#         row.sort()
# #         list.append(row[1:k+1])
#         list.append(row[1:k+1])
#     return list
# #     print(row.tolist().sort())


# In[31]:


# list_k1 = findkNeibour(1)


# In[32]:


# plt.rcParams["figure.figsize"] = [16,9]
# sorted(list_k1)
# plt.plot(list_k1)
# plt.show()


# In[33]:


fig = plt.figure()
plt.rcParams["figure.figsize"] = [15,9]



for i in range(3, 21):
    temp = sorted(row[i] for row in dist_mat_sorted)
    plt.plot(temp)

x_axis = fig.gca()
x_axis.set_yticks(np.arange(0, 180, 5))
    
# for i in range(0,200,10):
#     plt.axhline(i, color='r', linestyle='-')
plt.grid(True)
plt.show()


# In[49]:


eps = 3.5
density = 15
print(dist_mat_sorted.shape)


# In[50]:


def find_Core_Points(eps, density):
    visited_dict = {}
    unvisited_dict = {}
    for j in range(0, len(dist_mat_sorted)):
        count = 0
        if dist_mat_sorted[j][density]<=eps:            
            visited_dict[j] = 0
        else:
            unvisited_dict[j] = 0
    return list,visited_dict, unvisited_dict


# In[51]:


doc_core_list, visited_dict, unvisited_dict = find_Core_Points(eps, density)


# In[52]:


print(len(doc_core_list))
print(len(visited_dict))
print(len(unvisited_dict))
count1 = 0
count0 = 0
for i in doc_core_list:
    if i==0:
        count0+=1
    elif i==1:
        count1+=1
print("count of 1: ", count1)
print("count of 0: ", count0)


# In[53]:


def compute_euclidean_dist(a, b):
    return np.linalg.norm(a-b)


# In[54]:


cluster = defaultdict(list)


# In[55]:


# cluster_index = 0
# for k in sorted(visited_dict.keys()):
# # for i in range(0, len(dist_mat)):
#     if visited_dict.get(k)==0:
#         print("For Cluster Index: ", cluster_index, " for k : ", k, " value: ", visited_dict.get(k))
#         tempSet = [k]
#         for i in sorted(visited_dict.keys()):            
#             if visited_dict[i]==0:
#                 print("For core point: ", i)
#                 for j in range(i+1, len(dist_mat[i])):
#                     if visited_dict.get(j)==0:
#                         dist = compute_euclidean_dist(dist_mat[i][1], dist_mat[i][j])
# #                         print("dist", dist)
#                         if dist<=eps:
#                             tempSet.append(j)
#                             visited_dict[j] = 1
#         cluster[cluster_index].extend(tempSet)
#         cluster_index+=1
index1 = 0 


# In[56]:



def defineClusterPoints(k, flag=False):
    if flag:        
        temp_set = []
    else:
        temp_set = [k]
#     print(visited_dict.keys())
    for i in sorted(visited_dict.keys()):
#         print("before i: ", i , k,  visited_dict.get(i))
        if visited_dict.get(i)==0:
#             dist = compute_euclidean_dist(dist_mat[k][1], dist_mat[k][i])
            
#             if dist<=eps:
#             print("k:  ", k, "i:  ", i, " dist: ", dist[k][i], "eps: ", eps)
            if(dist_mat[k][i]<=eps):
                temp_set.append(i)
#                 index1=index1+1
#                 print("Total docs in cluster: ", index1)
                visited_dict[i] = 1
                print("value change: ", visited_dict[i])
                print("after i: ", i, k,   visited_dict.get(i))

#                 print(k, "  ", visited_dict[k])
    return temp_set         
#         cluster[cluster_index].extend(tempSet
#                     if visited_dict.get(j)==0:
#                         dist = compute_euclidean_dist(dist_mat[i][1], dist_mat[i][j])
# #                         print("dist", dist)
#                         if dist<=eps:
#                             tempSet.append(j)
#                             visited_dict[j] = 1


# In[57]:


def doesExistInCluster(k):
    for key in cluster.keys():
        if k in cluster.get(key):
            return True
    return False


# In[58]:


cluster_index = 0
# for m in sorted(visited_dict.keys()):
len_mat = len(dist_mat)
for k in sorted(visited_dict.keys()):
# for i in range(0, len(dist_mat)): 

#     print("Cluster Index ", cluster_index)
    if visited_dict.get(k)==0:        
        print("Non Visited ", k)
        temp_set = []
        temp_set = list(set(defineClusterPoints(k, False)))
        cluster[cluster_index] = temp_set
#         print ("Cluster Index: ", cluster_index, "   cluster length: ", len(cluster[cluster_index]))
        cluster_index+=1
#         visited_dict[k]=1
        print(k, "  ", visited_dict[k])
    else:
#         print("Visited ", k)
        for key in cluster.keys():
            if doesExistInCluster(k):                    
                temp_set = list(set(defineClusterPoints(k, True)))
                temp_set_old = cluster.get(key)
                temp_set_old.extend(temp_set)
                cluster[key] = list(set(temp_set_old))
    print ("k: ", k, "Cluster Index: ", cluster_index, "   cluster length: ", len(cluster[cluster_index-1]))
                    
#     print(visited_dict)
    


# In[59]:


print(len(cluster))
print(len(cluster[0]))
print(len(cluster[1]))
print(len(cluster[2]))


# In[60]:


sum = 0
for key in cluster.keys():
    print(len(set(cluster.get(key))))
    sum += len(cluster.get(key))
print(sum)
    


# In[61]:


from copy import deepcopy
clstr = deepcopy(cluster)
print(type(clstr))
print(len(clstr))
print(len(unvisited_dict))


# In[62]:


noise = []
# print(unvisited_dict.keys())
print(len(unvisited_dict.keys()))
ct = []


# In[63]:


for i in unvisited_dict.keys():    
    flag = False
    for key in clstr.keys():
        flag = False
        for j in clstr.get(key):
            dist = dist_mat[i][j]
            if dist<=eps:
                ct.append(i)
                temp_list = list(set(clstr.get(key)))
                temp_list.extend([i])            
                clstr[key] = list(set(temp_list))
                flag = True
                break;

        if flag is True:
#             flag = False
            print ("Boundary found_ flag true")
            break;
    if flag is False:
        noise.append(i)
        print ("Boundary not found_ flag false")        
        
        


# In[64]:


print(len(noise))
print(len(ct))


# In[65]:


clstr[len(clstr)] = noise
print(len(clstr))


# In[66]:


sum = 0
for key in clstr.keys():
#     print(len(set(cluster.get(key))))
#     print("cluster: ", len(cluster.get(key)), "clstr: ", len(clstr.get(key)))
    sum += len(clstr.get(key))
print(sum)


# In[67]:


output_dict = {}


# In[68]:


for key in clstr:
    for doc in clstr.get(key):
        output_dict[doc] = key 


# In[69]:


print(len(output_dict))


# In[70]:


def writeToFile(pred):
    with open('output/output1.dat','w+') as f:
        for p in sorted(output_dict.keys()):
            f.write(str(output_dict.get(p)+1)+"\n")


# In[71]:


writeToFile(output_dict)


# In[82]:


print(output_dict)


# In[72]:


from sklearn import metrics


# In[95]:


metrics.calinski_harabaz_score(dist_mat, output_dict.values())

