from typing import List, Tuple, Dict, Union
import os
import csv
from queue import PriorityQueue
from utils import *
import numpy as np
from sklearn.metrics.pairwise import nan_euclidean_distances

def csv_to_dict(path: str) -> dict:
    if not os.path.exists(path):
        raise Exception(f'The specified path {path} does not exist.')
    
    with open(path, 'r') as fp:
        reader = csv.reader(fp)
        titles = next(reader)
        data = {title: [] for title in titles}
        for row in reader:
            for i, title in enumerate(titles):
                data[title].append(row[i])
    
    return data

def load_question_meta(path: str) -> Dict[int, List[set]]:
    data = {}
    raw_data = csv_to_dict(os.path.join(path, 'question_meta.csv'))

    for q, s in zip(raw_data['question_id'], raw_data['subject_id']):
        subject_ids = set(map(int, s[1:-1].split(',')))
        data[int(q)] = subject_ids
    
    return data

def load_student_meta(path: str) -> Dict[int, int]:
    """Returns dictionary of student_id to year born."""
    data = {}
    raw_data = csv_to_dict(os.path.join(path, 'student_meta.csv'))

    for u, y in zip(raw_data['user_id'], raw_data['data_of_birth']):
        # Sometimes information not available
        if y == '':
            continue

        year = int(y.split('-')[0])
        data[int(u)] = year

    return data

def distance( 
    x,
    y,
    user_id_x, 
    user_id_y, 
    question_id, 
    question_meta, 
    student_meta,
    w1,
    w2,
) -> float:
    d = 0

    # Add nan euclidean distance
    d += nan_euclidean[user_id_x, user_id_y]

    # Add subject distance
    # Step 1. Get mutual question_ids
    question_ids = np.argwhere(np.logical_not(np.isnan(x)) & np.logical_not(np.isnan(y))).flatten()
    # Step 2. Compute fraction
    subjects = question_meta[question_id]
    count = 0
    total = 1
    for q in question_ids:
        count += len(subjects.intersection(question_meta[q]))
        total += len(question_meta[q])
    fraction = count / total

    if np.isnan(fraction):
        import pdb; pdb.set_trace()
    
    # Note w1 should be negative
    d += w1 * fraction

    # Add age distance
    if user_id_x in student_meta and user_id_y in student_meta:
        d += w2 * abs(student_meta[user_id_x] - student_meta[user_id_y])
    else:
        # 2.775490203357364 is average age difference
        d += w2 * 2.775490203357364
    
    return d

def predict(
    student_id, 
    question_id, 
    sparse_matrix, 
    question_meta, 
    student_meta, 
    k,
    w1,
    w2,
) -> int:
    """Predict if student_id answered question_id correctly."""
    # Step 1. Get rid of students who have not answered question_id from sparse_matrix
    # Store row of interest
    interest = sparse_matrix[student_id]
    answered = sparse_matrix[:, question_id]
    # user_ids are user_id of each row in new sparse_matrix
    user_ids = np.argwhere(np.logical_not(np.isnan(answered))).flatten()
    sparse_matrix = sparse_matrix[user_ids, :]

    # Step 2. Get k nearest neighbors
    pq = PriorityQueue()
    for i, row in enumerate(sparse_matrix):
        d = distance(
            interest,
            row,
            student_id,
            user_ids[i],
            question_id,
            question_meta,
            student_meta,
            w1,
            w2,
        )

        if np.isnan(d):
            continue
        
        # Pute -d since lowest have highest priority in pq
        pq.put((-d, i))
        if pq.qsize() > k:
            pq.get()
    closest_rows = [element[1] for element in pq.queue]
    vote = int(np.mean(sparse_matrix[closest_rows, question_id]) >= 0.5)

    return vote

# Global variable for nan_euclidean_distances
nan_euclidean = None

def main():
    sparse_matrix = load_train_sparse('../data').toarray()
    val_data = load_valid_csv('../data')
    test_data = load_public_test_csv('../data')
    question_meta = load_question_meta('../data')
    student_meta = load_student_meta('../data')

    # Precompute nan_euclidean
    global nan_euclidean
    nan_euclidean = nan_euclidean_distances(sparse_matrix)

    # Grid search
    # print('Initiating grid search...')
    # points = []
    # for k in [1, 6, 11, 16, 21]:
    #     for w1 in [0, -1, -2.5, -5, -7.5, -10]:
    #         for w2 in [0, 0.5, 1, 1.5, 2, 2.5]:
    #             print(f'k: {k}, w1: {w1}, w2: {w2}')
    #             predictions = []
    #             for i, q in enumerate(val_data['question_id']):
    #                 prediction = predict(val_data['user_id'][i], q, sparse_matrix, question_meta, student_meta, k, w1, w2)
    #                 predictions.append(prediction)
                
    #             acc = np.sum(np.array(predictions) == np.array(val_data['is_correct'])) / len(predictions)
    #             points.append((k, w1, w2, acc))
    #             print(f'Accuracy: {acc}')
    # point = None
    # m = 0
    # for p in points:
    #     if p[3] > m:
    #         m = p[3]
    #         point = p
    # print('Optimal point:', point)

    # Experiment
    # Default KNN with k = 11
    print('Default KNN with k = 11')
    k = 11
    w1 = 0
    w2 = 0
    predictions = []
    for i, q in enumerate(test_data['question_id']):
        prediction = predict(test_data['user_id'][i], q, sparse_matrix, question_meta, student_meta, k, w1, w2)
        predictions.append(prediction)
    
    acc = np.sum(np.array(predictions) == np.array(test_data['is_correct'])) / len(predictions)
    print(f'Test accuracy: {acc}')

    # Using optimal settings
    print('Using optimal settings')
    k = 11
    w1 = -5
    w2 = 0
    predictions = []
    for i, q in enumerate(test_data['question_id']):
        prediction = predict(test_data['user_id'][i], q, sparse_matrix, question_meta, student_meta, k, w1, w2)
        predictions.append(prediction)
    
    acc = np.sum(np.array(predictions) == np.array(test_data['is_correct'])) / len(predictions)
    print(f'Test accuracy: {acc}')

if __name__ == '__main__':
    main()