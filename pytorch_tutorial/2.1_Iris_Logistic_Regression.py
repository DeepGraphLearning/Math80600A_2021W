import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression


def set_seed(seed):
    np.random.seed(seed)
    return


def load_data():
    iris = datasets.load_iris()
    X = iris.data  # 150 * 4
    y = iris.target  # 150
    print('complete X: {}'.format(X.shape))
    print('complete y: {}'.format(y.shape))
    return X, y


if __name__ == '__main__':
    ########## Set-up seed for reproducibility ##########
    seed = 0
    set_seed(seed)

    ########## Load dataset ##########
    X, y = load_data()

    ########## Split dataset into train and test ##########
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    print('X_train', X_train.shape)
    print('y_train', y_train.shape)
    print('X_test', X_test.shape)
    print('y_test', y_test.shape)
    print()

    ########## Set model ##########
    model = LogisticRegression(C=1e5)
    model.fit(X_train, y_train)
    y_test_pred = model.predict(X_test)
    y_acc = np.sum(y_test_pred == y_test) / len(y_test_pred)
    print('accuracy: {}'.format(y_acc))

    W = model.coef_
    bias = model.intercept_
    print('W:', W)
    print('bias:\n', bias)
