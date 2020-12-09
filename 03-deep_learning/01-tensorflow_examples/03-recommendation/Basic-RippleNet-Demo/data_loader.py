import collections
import os
import numpy as np

def load_data(args):
    train_data,eval_data,test_data,user_history_dict = load_rating(args)
    n_entity,n_relation,kg = load_kg(args)
    ripple_set = get_ripple_set(args,kg,user_history_dict)
    return train_data,eval_data,test_data,n_entity,n_relation,ripple_set

def load_rating(args):
    print('reaing rating file ...')

    rating_file = 'data/ratings_final.txt'
    rating_np = np.loadtxt(rating_file,dtype=np.int32)

    return dataset_split(rating_np)


def dataset_split(rating_np):

    print('splitint dataset ...')
    eval_ratio = 0.2
    test_ratio = 0.2

    n_ratings = rating_np.shape[0]

    eval_indices = np.random.choice(n_ratings,size = int(n_ratings * eval_ratio),replace=False)
    left = set(range(n_ratings)) - set(eval_indices)
    test_indices = np.random.choice(list(left),size = int(n_ratings * test_ratio),replace=False)
    train_indices = list(left - set(test_indices))

    user_history_dict = dict()
    for i in train_indices:
        user = rating_np[i][0]
        item = rating_np[i][1]
        rating = rating_np[i][2]

        if rating == 1:
            if user not in user_history_dict:
                user_history_dict[user] = []
            user_history_dict[user].append(item)

    train_indices = [i for i in train_indices if rating_np[i][0] in user_history_dict]
    eval_indices = [i for i in eval_indices if rating_np[i][0] in user_history_dict]
    test_indices = [i for i in test_indices if rating_np[i][0] in user_history_dict]

    train_data = rating_np[train_indices]
    eval_data = rating_np[eval_indices]
    test_data = rating_np[test_indices]

    return train_data,eval_data,test_data,user_history_dict

def load_kg(args):
    print('reading KG file')
    kg_file = 'data/kg_final.txt'

    kg_np = np.loadtxt(kg_file,dtype=np.int32)

    n_entity = len(set(kg_np[:,0]) | set(kg_np[:,2]))
    n_relation = len(set(kg_np[:,1]))

    kg = construct_kg(kg_np)

    return n_entity,n_relation,kg

def construct_kg(kg_np):
    print('constructing knowledge graph ...')
    kg = collections.defaultdict(list)
    for head,relation,tail in kg_np:
        kg[head].append((tail,relation))
    return kg



def get_ripple_set(args,kg,user_history_dict):
    print('constructing ripple set')
    ripple_set = collections.defaultdict(list)

    for user in user_history_dict:
        for h in range(args.n_hop):
            memories_h = []
            memories_r = []
            memories_t = []

            if h == 0:
                tails_of_last_hop = user_history_dict[user]
            else:
                tails_of_last_hop = ripple_set[user][-1][2]

            for entity in tails_of_last_hop:
                for tail_and_relation in kg[entity]:
                    memories_h.append(entity)
                    memories_r.append(tail_and_relation[1])
                    memories_t.append(tail_and_relation[0])


            # if the current ripple set of the given user is empty, we simply copy the ripple set of the last hop here
            # this won't happen for h = 0, because only the items that appear in the KG have been selected
            # this only happens on 154 users in Book-Crossing dataset (since both BX dataset and the KG are sparse)
            if len(memories_h) == 0:
                ripple_set[user].append(ripple_set[user][-1])
            else:
                replace = len(memories_h) < args.n_memory
                indices = np.random.choice(len(memories_h),size = args.n_memory,replace= replace)
                memories_h = [memories_h[i] for i in indices]
                memories_r = [memories_r[i] for i in indices]
                memories_t = [memories_t[i] for i in indices]
                ripple_set[user].append((memories_h, memories_r, memories_t))

    return ripple_set



