import numpy as np
import matplotlib.pyplot as plt


seed = 47
np.random.seed(seed)

N = [100, 1000]

for training_examples in N:
    print("training examples being generated", training_examples)
    w0 = np.random.uniform(-0.25, 0.25)
    w1 = np.random.uniform(-1, 1)
    w2 = np.random.uniform(-1, 1)
    W = np.array([w0, w1, w2])
    print("optimal weights", W)
    X = np.random.uniform(low=[-1, -1], high=[1, 1], size=(training_examples, 2))

    S1, S0 = np.empty((0, 2)), np.empty((0, 2))
    Xy = np.empty((0, 2))

    for x in X:
        x1 = np.insert(x, 0, 1)
        if np.matmul(x1, np.transpose(W)) >= 0:
            S1 = np.append(S1, np.array([x]), axis=0)
            Xy = np.append(Xy, np.array([[x, np.array([1])]], dtype="object"), axis=0)
        else:
            S0 = np.append(S0, np.array([x]), axis=0)
            Xy = np.append(Xy, np.array([[x, np.array([0])]], dtype="object"), axis=0)

    x = np.linspace(-1, 1)
    plt.scatter(S1[:, 0], S1[:, 1], marker='X', label="S1")
    plt.scatter(S0[:, 0], S0[:, 1], marker='o', label="S0")
    plt.plot(x, -(w0+w1*x)/w2)
    plt.ylim(-1, 1)
    plt.xlim(-1, 1)
    plt.legend(loc="upper right")
    plt.savefig("scatterplot_{}.jpg".format(training_examples))
    plt.show()

    # training weights
    w01 = np.random.uniform(-1, 1)
    w11 = np.random.uniform(-1, 1)
    w21 = np.random.uniform(-1, 1)

    W1 = [w01, w11, w21]

    print("initial weights",W1)
    initial_misclassifaction_count = 0

    for x,y in Xy:
        x1 = np.insert(x, 0, 1)
        if np.matmul(x1,np.transpose(W1)) >= 0:
            if y != 1:
                initial_misclassifaction_count += 1
        else:
            if y != 0:
                initial_misclassifaction_count += 1
    print("Misclassification count before training",initial_misclassifaction_count)

    Learning_rate = [1, 10, 0.1]

    # Perceptron training algorithm for different learning rates
    for n in Learning_rate:
        W_n = W1
        epoch = 0
        converged_epoch = 0
        misclassification_count_per_epoch = []
        # used while loop as convergence epoch varies for each learning rate
        while True:
            misclassification_count = 0
            for x,y in Xy:
                x1 = np.insert(x, 0, 1)
                if np.matmul(x1,np.transpose(W_n)) >= 0:
                    if y != 1:
                        W_n = W_n - n * x1
                        misclassification_count += 1
                else:
                    if y != 0:
                        W_n = W_n + n * x1
                        misclassification_count += 1
            epoch += 1
            misclassification_count_per_epoch.append(misclassification_count)
            if misclassification_count == 0:
                converged_epoch = epoch
                print("converged at epoch", epoch)
                print("learning rate",n)
                print("weights at which algorithm converged",W_n)
                break

        epochs = list(range(1, converged_epoch+1))
        plt.plot(epochs, misclassification_count_per_epoch)
        plt.xlabel("epochs")
        plt.ylabel("misclassification_count")
        plt.title("Misclassification count with learning rate n = " + str(n) + '')
        plt.savefig("lr_{}_te_{}.jpg".format(n, training_examples))
        plt.show()


