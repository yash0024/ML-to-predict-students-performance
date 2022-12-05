from typing import List, Tuple

from utils import *
import numpy as np

from item_response import irt, evaluate, sigmoid

def resample(data: dict, n: int) -> dict:
    """Resample data without replacement.
    
    :param data: A dictionary {user_id: list, question_id: list,
    is_correct: list}
    :param n: A int declaring the number of samples to resample.
    :return: A dictionary {user_id: list, question_id: list,
    is_correct: list}
    """
    data = load_train_csv('../data')
    user_ids = np.unique(data['user_id'])
    resampled_user_ids = np.random.choice(user_ids, n, replace=False)
    resampled_data = {
        'user_id': [],
        'question_id': [],
        'is_correct': [],
    }
    for i, user_id in enumerate(data['user_id']):
        if user_id not in resampled_user_ids:
            continue
        resampled_data['user_id'].append(user_id)
        resampled_data['question_id'].append(data['question_id'][i])
        resampled_data['is_correct'].append(data['is_correct'][i])
    
    return resampled_data

def modify_val_data(val_data: dict, resampled_data: dict) -> dict:
    """Match validation data with resampled data.
    
    :param val_data: A dictionary {user_id: list, question_id: list,
    is_correct: list}
    :param resampled_data: A dictionary {user_id: list, question_id: list,
    is_correct: list}
    :return: A dictionary {user_id: list, question_id: list,
    is_correct: list}
    """
    user_ids = np.unique(resampled_data['user_id'])
    new_val_data = {
        'user_id': [],
        'question_id': [],
        'is_correct': [],
    }
    for i, user_id in enumerate(val_data['user_id']):
        if user_id not in user_ids:
            continue
        new_val_data['user_id'].append(user_id)
        new_val_data['question_id'].append(val_data['question_id'][i])
        new_val_data['is_correct'].append(val_data['is_correct'][i])

    return new_val_data
    
def ensemble_irt_predict(data: dict, models: List[Tuple[np.ndarray, np.ndarray, np.ndarray]])\
    -> np.ndarray:
    """Do an enesmble prediction.

    PRECONDITION: all 3 models must cover all the user_id in data!

    :param data: A dictionary {user_id: list, question_id: list, is_correct: list}
    :param models: A list of tuples (user_ids, theta, beta) representing models in ensemble
    :return: A numpy array of combined predictions
    """
    predictions = [[] for _ in range(len(models))]
    
    # Partly taken from item_response.py
    for i, q in enumerate(data["question_id"]):
        u = data["user_id"][i]
        for j, model in enumerate(models):
            user_ids, theta, beta = model

            # Important: how we handle missing users
            if u not in user_ids:
                predictions[j].append(np.nan)
                continue

            x = (theta[u] - beta[q]).sum()
            p_a = sigmoid(x)
            predictions[j].append(int(p_a >= 0.5))

    # Combine
    predictions = np.array(predictions)
    ensemble_prediction = (np.nanmean(predictions, axis=0) >= 0.5).astype(int)

    # Second return value for debugging
    return ensemble_prediction, predictions


def main():
    # Set seed to last 5 digits of a student number
    np.random.seed(77140)

    train_data = load_train_csv('../data')
    val_data = load_valid_csv('../data')
    test_data = load_public_test_csv('../data')
    N = np.unique(train_data['user_id']).size
    # We let resampled data contain 3/4 of the users of original
    n = int(N * 0.75)

    # We will have 3 base models
    base_models = []

    print('Training model 1.')
    # Model 1
    theta, beta, _ = irt(train_data, val_data, 0.01, 30)
    base_models.append([
        np.unique(train_data['user_id']),
        theta,
        beta,
    ])

    print('Training model 2.')
    # Model 2
    resampled_data_1 = resample(train_data, n)
    resampled_val_data_1 = modify_val_data(val_data, resampled_data_1)
    theta1, beta1, _ = irt(resampled_data_1, resampled_val_data_1, 0.01, 30)
    base_models.append([
        np.unique(resampled_data_1['user_id']),
        theta1,
        beta1,
    ])

    print('Training model 3.')
    # Model 3
    resampled_data_2 = resample(train_data, n)
    resampled_val_data_2 = modify_val_data(val_data, resampled_data_2)
    theta2, beta2, _ = irt(resampled_data_2, resampled_val_data_2, 0.01, 30)
    base_models.append([
        np.unique(resampled_data_2['user_id']),
        theta2,
        beta2,
    ])

    # Validation accuracy
    val_prediction, val_predictions = ensemble_irt_predict(val_data, base_models)
    val_accuracy = np.sum(val_prediction == np.array(val_data['is_correct'])) / len(val_prediction)

    print(f'Validation accuracy: {val_accuracy}')

    # Test accuracy
    test_prediction,  test_predictions = ensemble_irt_predict(test_data, base_models)
    test_accuracy = np.sum(test_prediction == np.array(test_data['is_correct'])) / len(test_prediction)

    print(f'Test accuracy: {test_accuracy}')

if __name__ == '__main__':
    main()
