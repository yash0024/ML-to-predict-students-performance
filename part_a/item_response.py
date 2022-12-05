from utils import *

import numpy as np
import matplotlib.pyplot as plt


def sigmoid(x):
    """ Apply sigmoid function.
    """
    return np.exp(x) / (1 + np.exp(x))


def neg_log_likelihood(data, theta, beta):
    """ Compute the negative log-likelihood.

    You may optionally replace the function arguments to receive a matrix.

    :param data: A dictionary {user_id: list, question_id: list,
    is_correct: list}
    :param theta: Vector
    :param beta: Vector
    :return: float
    """
    #####################################################################
    # TODO:                                                             #
    # Implement the function as described in the docstring.             #
    #####################################################################
    log_lklihood = 0.
    for i, q in enumerate(data["question_id"]):
        u = data['user_id'][i]
        ans = data['is_correct'][i]
        log_lklihood += ans * (theta[u] - beta[q]) - np.log(1 + np.exp(theta[u] - beta[q]))
    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################
    return -log_lklihood


def update_theta_beta(data, lr, theta, beta):
    """ Update theta and beta using gradient descent.

    You are using alternating gradient descent. Your update should look:
    for i in iterations ...
        theta <- new_theta
        beta <- new_beta

    You may optionally replace the function arguments to receive a matrix.

    :param data: A dictionary {user_id: list, question_id: list,
    is_correct: list}
    :param lr: float
    :param theta: Vector
    :param beta: Vector
    :return: tuple of vectors
    """
    #####################################################################
    # TODO:                                                             #
    # Implement the function as described in the docstring.             #
    #####################################################################
    d_theta, d_beta = np.zeros(np.shape(theta)), np.zeros(np.shape(beta))
    for i, q in enumerate(data["question_id"]):
        u = data['user_id'][i]
        ans = data['is_correct'][i]
        d_theta[u] += ans - sigmoid(theta[u] - beta[q])

    theta += lr * d_theta
    
    for i, q in enumerate(data['question_id']):
        u = data['user_id'][i]
        ans = data['is_correct'][i]
        d_beta[q] +=  - ans + sigmoid(theta[u] - beta[q])

    beta += lr * d_beta

    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################
    return theta, beta


def irt(data, val_data, lr, iterations):
    """ Train IRT model.

    You may optionally replace the function arguments to receive a matrix.

    :param data: A dictionary {user_id: list, question_id: list,
    is_correct: list}
    :param val_data: A dictionary {user_id: list, question_id: list,
    is_correct: list}
    :param lr: float
    :param iterations: int
    :return: (theta, beta, val_acc_lst)
    """
    # TODO: Initialize theta and beta.
    theta = np.ones((np.max(data['user_id']) + 1, 1))
    beta = np.ones((np.max(data['question_id']) + 1, 1))

    fig = plt.figure(figsize=(15,15))
    ax1 = fig.add_subplot(121)
    ax2 = fig.add_subplot(122)

    ax1.set(title = 'Training log-likelihood vs Iterations', xlabel = 'iteration', ylabel = 'training log-likelihood')
    ax2.set(title = 'Validation log-likelihood vs Iterations', xlabel = 'iteration', ylabel = 'validation log-likelihood')

    val_acc_lst = []
    log_train, log_val = [], []

    for i in range(iterations):
        neg_lld_train = neg_log_likelihood(data, theta = theta, beta = beta)
        neg_lld_val = neg_log_likelihood(val_data, theta = theta, beta = beta)
        log_train.append(-neg_lld_train)
        log_val.append(-neg_lld_val)
        score = evaluate(data=val_data, theta=theta, beta=beta)
        val_acc_lst.append(score)
        print("NLLK: {} \t Score: {}".format(neg_lld_train, score))
        theta, beta = update_theta_beta(data, lr, theta, beta)

    ax1.plot(range(iterations), log_train, color = 'blue')
    ax2.plot(range(iterations), log_val, color = 'blue')
    plt.show()

    # TODO: You may change the return values to achieve what you want.
    return theta, beta, val_acc_lst


def evaluate(data, theta, beta):
    """ Evaluate the model given data and return the accuracy.
    :param data: A dictionary {user_id: list, question_id: list,
    is_correct: list}

    :param theta: Vector
    :param beta: Vector
    :return: float
    """
    pred = []
    for i, q in enumerate(data["question_id"]):
        u = data["user_id"][i]
        x = (theta[u] - beta[q]).sum()
        p_a = sigmoid(x)
        pred.append(p_a >= 0.5)
    return np.sum((data["is_correct"] == np.array(pred))) \
           / len(data["is_correct"])


def main():
    train_data = load_train_csv("../data")
    # You may optionally use the sparse matrix.
    sparse_matrix = load_train_sparse("../data")
    val_data = load_valid_csv("../data")
    test_data = load_public_test_csv("../data")

    #####################################################################
    # TODO:                                                             #
    # Tune learning rate and number of iterations. With the implemented #
    # code, report the validation and test accuracy.                    #
    #####################################################################
    theta, beta, val_acc_list = irt(train_data, val_data, 0.01, 30)
    print('Valdiation accuracy = ', evaluate(val_data, theta, beta))
    print('Test accuracy = ', evaluate(test_data, theta, beta))
    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################

    #####################################################################
    # TODO:                                                             #
    # Implement part (d)                                                #
    #####################################################################
    num_questions = np.shape(beta)[0]
    questions = np.random.choice(num_questions, 3, replace = False)
    beta_1, beta_2, beta_3 = beta[questions[0]], beta[questions[1]], beta[questions[2]]
    thetas = np.arange(0, 4.2, 0.1)
    plot_1, plot_2, plot_3 = [], [], []
    for i in thetas:
        plot_1.append(sigmoid(i - beta_1))
        plot_2.append(sigmoid(i - beta_2))
        plot_3.append(sigmoid(i - beta_3))
    plt.title('p(c_ij) for 3 different questions using the trained beta, as a function of theta')
    plt.xlabel('theta')
    plt.ylabel('p(c_ij)')
    plt.plot(thetas, plot_1, color = 'r', label = 'Question ' + str(questions[0]))
    plt.plot(thetas, plot_2, color = 'g', label = 'Question ' + str(questions[1]))
    plt.plot(thetas, plot_3, color = 'b', label = 'Question ' + str(questions[2]))
    plt.legend()
    plt.show()
    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################


if __name__ == "__main__":
    main()