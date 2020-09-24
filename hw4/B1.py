
import csv
import numpy as np

def load_data():
    data = []
    with open('u.data') as csvfile:
        spamreader = csv.reader(csvfile, delimiter='\t') 
        for row in spamreader:
            data.append([int(row[0])-1, int(row[1])-1, int(row[2])]) 

    data = np.array(data)

    num_observations = len(data) # num_observations = 100,000
    num_users = max(data[:,0])+1 # num_users = 943, indexed 0,...,942
    num_items = max(data[:,1])+1 # num_items = 1682 indexed 0,...,1681

    np.random.seed(1)
    num_train = int(0.8*num_observations)
    perm = np.random.permutation(data.shape[0]) 
    train = data[perm[0:num_train],:]
    test = data[perm[num_train::],:]
    return train, test