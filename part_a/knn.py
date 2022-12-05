from sklearn.impute import KNNImputer
from utils import *
from matplotlib import pyplot as plt
import numpy as np


def knn_impute_by_user(matrix, valid_data, k):
    """ Fill in the missing values using k-Nearest Neighbors based on
    student similarity. Return the accuracy on valid_data.

    See https://scikit-learn.org/stable/modules/generated/sklearn.
    impute.KNNImputer.html for details.

    :param matrix: 2D sparse matrix
    :param valid_data: A dictionary {user_id: list, question_id: list,
    is_correct: list}
    :param k: int
    :return: float
    """
    nbrs = KNNImputer(n_neighbors=k)
    # We use NaN-Euclidean distance measure.
    mat = nbrs.fit_transform(matrix)
    acc = sparse_matrix_evaluate(valid_data, mat)
    print("Validation Accuracy: {}".format(acc))
    return acc


def knn_impute_by_item(matrix, valid_data, k):
    """ Fill in the missing values using k-Nearest Neighbors based on
    question similarity. Return the accuracy on valid_data.

    :param matrix: 2D sparse matrix
    :param valid_data: A dictionary {user_id: list, question_id: list,
    is_correct: list}
    :param k: int
    :return: float
    """
    #####################################################################
    # TODO:                                                             #
    # Implement the function as described in the docstring.             #
    #####################################################################
    matrix = matrix.T
    nbrs = KNNImputer(n_neighbors=k)
    # We use NaN-Euclidean distance measure.
    mat = nbrs.fit_transform(matrix).T
    acc = sparse_matrix_evaluate(valid_data, mat)
    print("Validation Accuracy: {}".format(acc))
    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################
    return acc


def main():
    sparse_matrix = load_train_sparse('../data').toarray()
    val_data = load_valid_csv('../data')
    test_data = load_public_test_csv('../data')

    print("Sparse matrix:")
    print(sparse_matrix)
    print("Shape of sparse matrix:")
    print(sparse_matrix.shape)

    #####################################################################
    # TODO:                                                             #
    # Compute the validation accuracy for each k. Then pick k* with     #
    # the best performance and report the test accuracy with the        #
    # chosen k*.                                                        #
    #####################################################################

    # By user
    print('By user')

    ks = [1, 6, 11, 16, 21, 26]
    accs = []
    for k in ks:
        acc = knn_impute_by_user(sparse_matrix, val_data, k)
        accs.append(acc)
    accs = np.array(accs)
    fig1, ax1 = plt.subplots(figsize=(10, 5))
    ax1.plot(ks, accs * 100)
    ax1.set_xlabel('k')
    ax1.set_ylabel('Validation Accuracy (%)')
    ax1.set_title('Validation Accuracy vs k')
    fig1.savefig('knn_impute_by_user.png')
    k_star = ks[np.argmax(accs)]
    print(f'k* = {k_star}')
    test_acc = knn_impute_by_user(sparse_matrix, test_data, k_star)
    print(f'Test accuracy: {test_acc}')

    # By item
    print('By item')

    accs = []
    for k in ks:
        acc = knn_impute_by_item(sparse_matrix, val_data, k)
        accs.append(acc)
    accs = np.array(accs)
    fig2, ax2 = plt.subplots(figsize=(10, 5))
    ax2.plot(ks, accs * 100)
    ax2.set_xlabel('k')
    ax2.set_ylabel('Validation Accuracy (%)')
    ax2.set_title('Validation Accuracy vs k')
    fig2.savefig('knn_impute_by_item.png')
    k_star = ks[np.argmax(accs)]
    print(f'k* = {k_star}')
    test_acc = knn_impute_by_item(sparse_matrix, test_data, k_star)
    print(f'Test accuracy: {test_acc}')
    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################


if __name__ == "__main__":
    main()
