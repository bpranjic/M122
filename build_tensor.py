import numpy as np
import pandas as pd

def build_tensor(features, triplets_file, gen_labels=False):
    triplets = pd.read_csv(triplets_file, delim_whitespace=True, header=None, names=["A", "B", "C"])
    train_tensors = []
    labels = []
    num_triplets = len(triplets)

    for i in range(num_triplets):
        triplet = triplets.iloc[i]
        A, B, C = triplet['A'], triplet['B'], triplet['C']
        tensorA = features[A]
        tensorB = features[B]
        tensorC = features[C]
        triplet_tensor = np.concatenate((tensorA, tensorB, tensorC), axis=-1)
        if(gen_labels):
            invert_tensor = np.concatenate((tensorA, tensorC, tensorB), axis=-1)
            train_tensors.append(triplet_tensor)
            labels.append(1)
            train_tensors.append(invert_tensor)
            labels.append(0)
        else:
            train_tensors.append(triplet_tensor)
    
    train_tensors = np.array(train_tensors)
    if(gen_labels):
        labels = np.array(labels)
        return train_tensors, labels
    else:
        return train_tensors