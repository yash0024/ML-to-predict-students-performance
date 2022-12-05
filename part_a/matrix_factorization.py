from utils import *
from scipy.linalg import sqrtm
from matplotlib import pyplot as plt

import numpy as np


def svd_reconstruct(matrix, k):
    """ Given the matrix, perform singular value decomposition
    to reconstruct the matrix.

    :param matrix: 2D sparse matrix
    :param k: int
    :return: 2D matrix
    """
    # First, you need to fill in the missing values (NaN) to perform SVD.
    # Fill in the missing values using the average on the current item.
    # Note that there are many options to do fill in the
    # missing values (e.g. fill with 0).
    new_matrix = matrix.copy()
    mask = np.isnan(new_matrix)
    masked_matrix = np.ma.masked_array(new_matrix, mask)
    item_means = np.mean(masked_matrix, axis=0)
    new_matrix = masked_matrix.filled(item_means)

    # Next, compute the average and subtract it.
    item_means = np.mean(new_matrix, axis=0)
    mu = np.tile(item_means, (new_matrix.shape[0], 1))
    new_matrix = new_matrix - mu

    # Perform SVD.
    Q, s, Ut = np.linalg.svd(new_matrix, full_matrices=False)
    s = np.diag(s)

    # Choose top k eigenvalues.
    s = s[0:k, 0:k]
    Q = Q[:, 0:k]
    Ut = Ut[0:k, :]
    s_root = sqrtm(s)

    # Reconstruct the matrix.
    reconst_matrix = np.dot(np.dot(Q, s_root), np.dot(s_root, Ut))
    reconst_matrix = reconst_matrix + mu
    return np.array(reconst_matrix)


def squared_error_loss(data, u, z):
    """ Return the squared-error-loss given the data.
    :param data: A dictionary {user_id: list, question_id: list,
    is_correct: list}
    :param u: 2D matrix
    :param z: 2D matrix
    :return: float
    """
    A = u @ z.transpose()
    arr = A[data["user_id"], data["question_id"]]
    loss = np.sum((data["is_correct"] - arr) ** 2)
    return 0.5 * loss


def squared_error_loss_original(data, u, z):
    """ Return the squared-error-loss given the data.
    :param data: A dictionary {user_id: list, question_id: list,
    is_correct: list}
    :param u: 2D matrix
    :param z: 2D matrix
    :return: float
    """
    loss = 0
    for i, q in enumerate(data["question_id"]):
        loss += (data["is_correct"][i]
                 - np.sum(u[data["user_id"][i]] * z[q])) ** 2.
    return 0.5 * loss


def update_u_z(train_data, lr, u, z):
    """ Return the updated U and Z after applying
    stochastic gradient descent for matrix completion.

    :param train_data: A dictionary {user_id: list, question_id: list,
    is_correct: list}
    :param lr: float
    :param u: 2D matrix
    :param z: 2D matrix
    :return: (u, z)
    """
    #####################################################################
    # TODO:                                                             #
    # Implement the function as described in the docstring.             #
    #####################################################################
    # Randomly select a pair (user_id, question_id).
    i = \
        np.random.choice(len(train_data["question_id"]), 1)[0]
    c = train_data["is_correct"][i]
    n = train_data["user_id"][i]
    q = train_data["question_id"][i]

    u[n] -= lr * -(c - np.sum(u[n] * z[q])) * z[q]
    z[q] -= lr * -(c - np.sum(u[n] * z[q])) * u[n]
    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################
    return u, z


def als(train_data, k, lr, num_iteration):
    """ Performs ALS algorithm, here we use the iterative solution - SGD
    rather than the direct solution.

    :param train_data: A dictionary {user_id: list, question_id: list,
    is_correct: list}
    :param k: int
    :param lr: float
    :param num_iteration: int
    :return: 2D reconstructed Matrix.
    """
    # Initialize u and z
    u = np.random.uniform(low=0, high=1 / np.sqrt(k),
                          size=(len(set(train_data["user_id"])), k))
    z = np.random.uniform(low=0, high=1 / np.sqrt(k),
                          size=(len(set(train_data["question_id"])), k))
    # rows used to be len(set(train_data["user_id"]))
    #####################################################################
    # TODO:                                                             #
    # Implement the function as described in the docstring.             #
    #####################################################################
    # val_data = load_valid_csv("../data")
    # iterations = np.arange(num_iteration)
    # train_losses = []
    # val_losses = []

    for i in range(num_iteration):
        u, z = update_u_z(train_data, lr, u, z)
        # train_losses.append(squared_error_loss(train_data, u, z))
        # val_losses.append(squared_error_loss(val_data, u, z))

    mat = np.matmul(u, z.transpose())


    # fig1, ax1 = plt.subplots(figsize=(10, 5))
    # ax1.plot(iterations, train_losses)
    # ax1.set_xlabel('Iteration #')
    # ax1.set_ylabel('Training Loss')
    # ax1.set_title('Iteration # vs. Training Loss')
    # fig1.savefig('training_loss.png')

    # fig2, ax2 = plt.subplots(figsize=(10, 5))
    # ax2.plot(iterations, val_losses)
    # ax2.set_xlabel('Iteration #')
    # ax2.set_ylabel('Validation Loss')
    # ax2.set_title('Iteration # vs. Validation Loss')
    # fig2.savefig('validation_loss.png')

    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################
    return mat


def main():
    train_matrix = load_train_sparse("../data").toarray()
    train_data = load_train_csv("../data")
    val_data = load_valid_csv("../data")
    test_data = load_public_test_csv("../data")

    #####################################################################
    # TODO:                                                             #
    # (SVD) Try out at least 5 different k and select the best k        #
    # using the validation set.                                         #
    #####################################################################
    # (a)

    best_k = 0
    best_accuracy = 0
    best_matrix = []
    for k in [10, 20, 30, 50, 75, 100, 200]:
        matrix = svd_reconstruct(train_matrix, k)
        accuracy = sparse_matrix_evaluate(val_data, matrix)
        if accuracy > best_accuracy:
            best_k = k
            best_accuracy = accuracy
            best_matrix = matrix
    print("The optimal value of k is " + str(best_k) +
          ". Its validation and test accuracies are " + str(best_accuracy) +
          " and " + str(sparse_matrix_evaluate(test_data, best_matrix)))
    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################

    #####################################################################
    # TODO:                                                             #
    # (ALS) Try out at least 5 different k and select the best k        #
    # using the validation set.                                         #
    #####################################################################
    # (d)

    best_k = 0
    best_lr = 0
    best_iterations = 0
    best_accuracy = 0
    """
    for k in [10, 20, 30, 50, 75, 100]:
        for lr in [0.001, 0.005, 0.01, 0.05, 0.1]:
            for num_iterations in [10000, 50000, 100000, 250000]:
                matrix = als(train_data, k, lr, num_iterations)
                accuracy = sparse_matrix_evaluate(val_data, matrix)
                if accuracy > best_accuracy:
                    best_k = k
                    best_lr = lr
                    best_iterations = num_iterations
                    best_accuracy = accuracy
    print(best_k)
    print(best_lr)
    print(best_iterations)
    print(best_accuracy)
    """
    # 30, 0.05, 100,000, 0.7019


    # (e)
    matrix = als(train_data, 30, 0.05, 100000)
    val_acc = sparse_matrix_evaluate(val_data, matrix)
    test_acc = sparse_matrix_evaluate(test_data, matrix)
    print("The final validation and test accuracies are " +
          str(val_acc) + " and " + str(test_acc))

    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################


if __name__ == "__main__":
    main()