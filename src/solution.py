from scipy import io
import enum
import numpy

train_data_file_path = '../data/train_data.mat'
test_data_file_path = '../data/test_data.mat'


# create an enumeration for identifying classes
class Class(enum.Enum):
    class0 = 3
    class1 = 7


def load_data():
    train_data_class0 = extract_class_data(train_data_file_path, Class.class0)
    train_data_class1 = extract_class_data(train_data_file_path, Class.class1)
    test_data_class0 = extract_class_data(test_data_file_path, Class.class0)
    test_data_class1 = extract_class_data(test_data_file_path, Class.class1)
    return train_data_class0, train_data_class1, test_data_class0, test_data_class1


# extract the data relevant to specified class
def extract_class_data(file, class_name):
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
    return io.loadmat(file)


def get_features(data_class):
    x_vector = numpy.empty(shape=(len(data_class), 2), dtype=float)
    for x in range(len(data_class)):
        mean = get_mean(data_class[x])
        variance = get_std(data_class[x])
        x_vector[x, :] = [mean, variance]
    return x_vector


def get_mean(array):
    return numpy.mean(array)


def get_std(array):
    return numpy.std(array)


def get_y_vector(x_vector, m0, s0, m1, s1):
    return get_normalized_data(x_vector, m0, s0, m1, s1)


def get_x_vector_params_train(x_vector_train):
    x0 = x_vector_train[:, 0]
    x1 = x_vector_train[:, 1]
    m0 = get_mean(x0)
    s0 = get_std(x0)
    m1 = get_mean(x1)
    s1 = get_std(x1)
    return m0, s0, m1, s1


def get_normalized_data(x_vector, m0, s0, m1, s1):
    y_vector = numpy.empty(shape=(len(x_vector), 2), dtype=float)
    for x in range(len(x_vector)):
        if s0 != 0:
            y1_normalized = (x_vector[x][0] - m0) / s0
        else:
            raise ZeroDivisionError
        if s1 != 0:
            y2_normalized = (x_vector[x][1] - m1) / s1
        else:
            raise ZeroDivisionError
        y_vector[x, :] = [y1_normalized, y2_normalized]
    return y_vector


def print_debug(description, vector):
    assert isinstance(description, str)
    print(description + ":")
    print(len(vector))


def main():
    # load data
    train_data_class0, train_data_class1, test_data_class0, test_data_class1 = load_data()

    # Task 1: Feature extraction and normalization

    # get X Vectors for training data
    x_vector_class0_train = get_features(train_data_class0)
    x_vector_class1_train = get_features(train_data_class1)

    # find M1, S1, M2, S2
    m00, s00, m10, s10 = get_x_vector_params_train(x_vector_class0_train)
    m01, s01, m11, s11 = get_x_vector_params_train(x_vector_class0_train)

    # get Y Vectors for training data
    y_vector_class0_train = get_y_vector(x_vector_class0_train, m00, s00, m10, s10)
    y_vector_class1_train = get_y_vector(x_vector_class1_train,  m01, s01, m11, s11)

    # for debugging
    print_debug("Size of Y Vector Class 0 Training", y_vector_class0_train)
    print_debug("Size of Y Vector Class 1 Training", y_vector_class1_train)

    # get X Vectors for testing data
    x_vector_class0_test = get_features(test_data_class0)
    x_vector_class1_test = get_features(test_data_class1)

    # get Y Vectors for testing data
    y_vector_class0_test = get_y_vector(x_vector_class0_test, m00, s00, m10, s10)
    y_vector_class1_test = get_y_vector(x_vector_class1_test, m01, s01, m11, s11)

    # for debugging
    print_debug("Size of Y Vector Class 0 Testing", y_vector_class0_test)
    print_debug("Size of Y Vector Class 1 Testing", y_vector_class1_test)


# Using the special variable
# __name__
if __name__ == "__main__":
    main()
