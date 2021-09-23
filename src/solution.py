# Python 3.8.10
from scipy import io
import enum
import numpy

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
    for x in range(len(x_vector)):
        mean = get_mean(data_class[x])
        variance = get_std(data_class[x])
        x_vector[x, :] = [mean, variance]
    return x_vector


def get_mean(array):
    assert isinstance(array, numpy.ndarray)
    return numpy.mean(array)


def get_std(array):
    assert isinstance(array,  numpy.ndarray)
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
    assert isinstance(m1, float)
    assert isinstance(s1, float)
    y_vector = numpy.empty(shape=(len(vector), 2), dtype=float)
    for x in range(len(y_vector)):
        if s0 != 0:
            y1_normalized = (vector[x][0] - m0) / s0
        else:
            raise ZeroDivisionError
        if s1 != 0:
            y2_normalized = (vector[x][1] - m1) / s1
        else:
            raise ZeroDivisionError
        y_vector[x, :] = [y1_normalized, y2_normalized]
    return y_vector


# print dimension on console for debugging
def print_dimension(vector_name, vector):
    assert isinstance(vector_name, str)
    assert isinstance(vector, numpy.ndarray)
    print("Dimension of " + vector_name + ": ", vector.shape)


# endregion

def main():
    # load data
    train_data_class0, train_data_class1, test_data_class0, test_data_class1 = load_data()

    # Task 1: Feature extraction and normalization

    # get class0 and class1 X Vectors for training data
    x_vector_class0_train = get_features(train_data_class0)
    x_vector_class1_train = get_features(train_data_class1)

    # find M1, S1, M2, S2 for class0 and class1
    m00, s00, m10, s10 = get_x_vector_params_train(x_vector_class0_train)
    m01, s01, m11, s11 = get_x_vector_params_train(x_vector_class0_train)

    # get class0 and class1 Y Vectors for training data
    y_vector_class0_train = get_normalized_data(x_vector_class0_train, m00, s00, m10, s10)
    y_vector_class1_train = get_normalized_data(x_vector_class1_train, m01, s01, m11, s11)

    # for debugging
    print_dimension("Y Vector Class 0 Training", y_vector_class0_train)
    print_dimension("Y Vector Class 1 Training", y_vector_class1_train)

    # get lass0 and class1 X Vectors for testing data
    x_vector_class0_test = get_features(test_data_class0)
    x_vector_class1_test = get_features(test_data_class1)

    # get class0 and class1 Y Vectors for testing data
    y_vector_class0_test = get_normalized_data(x_vector_class0_test, m00, s00, m10, s10)
    y_vector_class1_test = get_normalized_data(x_vector_class1_test, m01, s01, m11, s11)

    # for debugging
    print_dimension("Y Vector Class 0 Testing", y_vector_class0_test)
    print_dimension("Y Vector Class 1 Testing", y_vector_class1_test)


# Using the special variable
# __name__
if __name__ == "__main__":
    main()
