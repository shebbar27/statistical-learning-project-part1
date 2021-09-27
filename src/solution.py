# Python 3.8.10
import enum
import matplotlib.pyplot as plotter
import numpy
from scipy import io
from scipy.stats import multivariate_normal
from typing import List

# file paths for loading training and testing sample data
train_data_file_path = '../data/train_data.mat'
test_data_file_path = '../data/test_data.mat'


# create an enumeration for identifying classes
class Class(enum.Enum):
    class0 = 3
    class1 = 7


# region helper methods for task1

def load_data():
    train_data_class0 = extract_class_data(train_data_file_path, Class.class0)
    train_data_class1 = extract_class_data(train_data_file_path, Class.class1)
    test_data_class0 = extract_class_data(test_data_file_path, Class.class0)
    test_data_class1 = extract_class_data(test_data_file_path, Class.class1)
    return train_data_class0, train_data_class1, test_data_class0, test_data_class1


# extract the data relevant to specified class from the file
def extract_class_data(file, class_name):
    assert isinstance(file, str)
    assert isinstance(class_name, Class)
    train_set = load_data_from_mat_file(file)
    train_set_label = train_set['label']
    train_set_data = train_set['data']
    train_data_class = []
    assert isinstance(train_set_label.size, int)
    for x in range(train_set_label[0].size):
        if train_set_label[0][x] == class_name.value:
            train_data_class.append(train_set_data[x])
    return train_data_class


# load data from .mat file
def load_data_from_mat_file(file):
    assert isinstance(file, str)
    return io.loadmat(file)


# get features from raw image samples (i.e mean and standard deviation of each image)
def get_features(data_class):
    assert isinstance(data_class, list)
    x_vector = numpy.empty(shape=(len(data_class), 2), dtype=float)
    for x in range(len(data_class)):
        mean = get_mean(data_class[x])
        variance = get_std(data_class[x])
        x_vector[x, :] = [mean, variance]
    return x_vector


def get_mean(array):
    assert isinstance(array, numpy.ndarray)
    return numpy.mean(array)


def get_std(array):
    assert isinstance(array, numpy.ndarray)
    return numpy.std(array)


# calculate mean & standard deviations for features in x_vector
def get_x_vector_params_train(x_vector):
    assert isinstance(x_vector, numpy.ndarray)
    x0 = x_vector[:, 0]
    x1 = x_vector[:, 1]
    m0 = get_mean(x0)
    s0 = get_std(x0)
    m1 = get_mean(x1)
    s1 = get_std(x1)
    return m0, s0, m1, s1


# calculate normalized values for all the elements of the given vector with 2 features
def get_normalized_data(vector, m0, s0, m1, s1):
    assert isinstance(vector, numpy.ndarray)
    assert vector.shape[1] == 2
    assert isinstance(m0, float)
    assert isinstance(s0, float)
    assert s0 != 0
    assert isinstance(m1, float)
    assert isinstance(s1, float)
    assert s1 != 0
    y_vector = numpy.empty(shape=(len(vector), 2), dtype=float)
    for x in range(len(y_vector)):
        y1_normalized = (vector[x][0] - m0) / s0
        y2_normalized = (vector[x][1] - m1) / s1
        y_vector[x, :] = [y1_normalized, y2_normalized]
    return y_vector


# print dimension on console for debugging
def print_dimension(vector_name, vector):
    assert isinstance(vector_name, str)
    assert isinstance(vector, numpy.ndarray)
    print("Dimension of " + vector_name + ": ", vector.shape)


# plot a list of 2-d vectors
def plot_2d_vectors(vectors, title="Figure1", x_label="x", y_label="y"):
    assert isinstance(vectors, List)
    assert isinstance(vectors[0], numpy.ndarray)
    assert isinstance(vectors[1], numpy.ndarray)
    assert vectors[0].shape[1] == 2
    assert vectors[1].shape[1] == 2
    assert isinstance(title, str)
    assert isinstance(x_label, str)
    assert isinstance(y_label, str)
    colors = ['orangered', 'forestgreen']
    for i in range(len(vectors)):
        class_label = get_axis_label(i)
        plotter.scatter(x=vectors[i][:, 0], y=vectors[i][:, 1], s=1, c=colors[i % 2], label=class_label)
        plotter.legend(loc='upper left')
    plotter.xlabel(x_label)
    plotter.ylabel(y_label)
    plotter.title(title)
    plotter.show()


def get_axis_label(class_num):
    if class_num == 0:
        return f'Class 0 (Digit {Class.class0.value})'
    if class_num == 1:
        return f'Class 0 (Digit {Class.class1.value})'
    return ''


# endregion

# region helper methods for task2

# compute MLE parameter1, mu assuming the given distribution is normal with unknown mean and variance
def compute_mu(vector):
    assert isinstance(vector, numpy.ndarray)
    assert vector.shape[0] != 0
    assert vector.shape[1] == 2
    n = len(vector)
    return 1 / n * numpy.array(
        [[numpy.sum(vector[:, 0])],
         [numpy.sum(vector[:, 1])]],
        dtype=float)


# compute MLE parameter2, sigma assuming given distribution is normal with unknown mean and variance
def compute_sigma(vector, mu):
    assert isinstance(vector, numpy.ndarray)
    assert vector.shape[0] != 0
    assert vector.shape[1] == 2
    assert isinstance(mu, numpy.ndarray)
    assert mu.shape == (2, 1)
    n = len(vector)
    vector[:, 0] -= mu[0][0]
    vector[:, 1] -= mu[1][0]
    return 1 / n * numpy.dot(vector.transpose(), vector)


# endregion

# region helper methods for task3

def calculate_error(mu0, sigma0, priori0, mu1, sigma1, priori1, y):
    assert isinstance(mu0, numpy.ndarray)
    assert mu0.shape == (2, 1)
    assert isinstance(sigma0, numpy.ndarray)
    assert sigma0.shape == (2, 2)
    assert isinstance(mu1, numpy.ndarray)
    assert mu1.shape == (2, 1)
    assert isinstance(sigma1, numpy.ndarray)
    assert sigma1.shape == (2, 2)
    assert isinstance(priori0, float)
    assert isinstance(priori1, float)
    assert y.shape[0] != 0
    assert y.shape[1] == 2
    dist0 = multivariate_normal(mean=mu0.flatten(), cov=sigma0)
    dist1 = multivariate_normal(mean=mu1.flatten(), cov=sigma1)
    probs = []
    for x in y:
        prob0 = dist0.pdf(x) * priori0
        prob1 = dist1.pdf(x) * priori1
        probs.append(min(prob0, prob1))
    return 1 / len(y) * numpy.sum(probs)


# endregion


def main():
    # load data
    train_data_class0, train_data_class1, test_data_class0, test_data_class1 = load_data()

    # region Task 1: Feature extraction and normalization

    # get class0 and class1 X Vectors for training data
    x_vector_class0_train = get_features(train_data_class0)
    x_vector_class1_train = get_features(train_data_class1)

    # find M1, S1, M2, S2 for class0 and class1
    m00, s00, m10, s10 = get_x_vector_params_train(x_vector_class0_train)
    m01, s01, m11, s11 = get_x_vector_params_train(x_vector_class0_train)

    # get class0 and class1 Y Vectors for training data
    y_vector_class0_train = get_normalized_data(x_vector_class0_train, m00, s00, m10, s10)
    y_vector_class1_train = get_normalized_data(x_vector_class1_train, m01, s01, m11, s11)
    plot_2d_vectors([y_vector_class0_train, y_vector_class1_train],
                    'Training Data', 'Y1: Mean (Normalized)', 'Y2: Standard Deviation (Normalized)')

    # for debugging
    # print_dimension('Y Vector Class 0 Training', y_vector_class0_train)
    # print_dimension('Y Vector Class 1 Training', y_vector_class1_train)

    # get lass0 and class1 X Vectors for testing data
    x_vector_class0_test = get_features(test_data_class0)
    x_vector_class1_test = get_features(test_data_class1)

    # get class0 and class1 Y Vectors for testing data
    y_vector_class0_test = get_normalized_data(x_vector_class0_test, m00, s00, m10, s10)
    y_vector_class1_test = get_normalized_data(x_vector_class1_test, m01, s01, m11, s11)
    plot_2d_vectors([y_vector_class0_test, y_vector_class1_test],
                    'Testing Data', 'Y1: Mean (Normalized)', 'Y2: Standard Deviation (Normalized)')

    # for debugging
    # print_dimension('Y Vector Class 0 Testing', y_vector_class0_test)
    # print_dimension('Y Vector Class 1 Testing', y_vector_class1_test)

    # endregion

    # region Task 2: Density estimation
    mu_class0_train = compute_mu(y_vector_class0_train)
    mu_class1_train = compute_mu(y_vector_class1_train)
    sigma_class0_train = compute_sigma(y_vector_class0_train, mu_class0_train)
    sigma_class1_train = compute_sigma(y_vector_class1_train, mu_class1_train)

    # for debugging
    print('Mu for Class0 Training Data:\n', mu_class0_train)
    print('Sigma for Class0 Training Data:\n', sigma_class0_train)
    print('Mu for Class1 Training Data:\n', mu_class1_train)
    print('Sigma for Class1 Training Data:\n', sigma_class1_train)

    # endregion

    # region Task 3: Bayesian Decision Theory for optimal classification

    all_class_train_data = numpy.append(y_vector_class0_train, y_vector_class1_train, axis=0)
    all_class_test_data = numpy.append(y_vector_class0_test, y_vector_class1_test, axis=0)

    # Case 1: priori(class0) = priori(class1) = 0.5
    priori_class0 = 0.5
    priori_class1 = 0.5
    prob_error_case0_train = calculate_error(
        mu_class0_train,
        sigma_class0_train,
        priori_class0,
        mu_class1_train,
        sigma_class1_train,
        priori_class1,
        all_class_train_data
    )

    prob_error_case0_test = calculate_error(
        mu_class0_train,
        sigma_class0_train,
        priori_class0,
        mu_class1_train,
        sigma_class1_train,
        priori_class1,
        all_class_test_data
    )

    # for debugging
    print(f'Case0: Probability of Error for Training Data = {prob_error_case0_train:.5f}')
    print(f'Case0: Probability of Error for Testing Data = {prob_error_case0_test:.5f}')

    # Case 2: priori(class0) = 0.3 and priori(class1) = 0.7
    priori_class0 = 0.3
    priori_class1 = 0.7
    prob_error_case1_train = calculate_error(
        mu_class0_train,
        sigma_class0_train,
        priori_class0,
        mu_class1_train,
        sigma_class1_train,
        priori_class1,
        all_class_train_data
    )

    prob_error_case1_test = calculate_error(
        mu_class0_train,
        sigma_class0_train,
        priori_class0,
        mu_class1_train,
        sigma_class1_train,
        priori_class1,
        all_class_test_data
    )

    # for debugging
    print(f'Case1: Probability of Error for Training Data = {prob_error_case1_train:.5f}')
    print(f'Case1: Probability of Error for Testing Data = {prob_error_case1_test:.5f}')

    # endregion


# Using the special variable
# __name__
if __name__ == "__main__":
    main()
