from utils import *
from cost_utils import *
from model import *
from dense_utils import *
from conv_utils import *
from visu_utils import *

# Load data
X_train, Y_train, X_test = load_and_process_data(n_elem=10)

print("X_train shape: {}".format(X_train.shape))
print("Y_train shape: {}".format(Y_train.shape))
print("X_test shape: {}".format(X_test.shape))

# Plot random images
#plot_random_image(X_train, 10, Y_train)

def test_dense_model(X_train, Y_train):
    flat1 = FlattenLayer(prev_layer=None, input_shape=X_train.shape)
    dense1 = DenseLayer(prev_layer=flat1, units=20, activation='sigmoid')
    dense2 = DenseLayer(prev_layer=dense1, units=10, activation='sigmoid')
    layers = [flat1, dense1, dense2]

    model = Model(layers)
    print('Model input shape: {}'.format(model.input_shape))

    model.check_gradients(X_train, Y_train)
    costs = model.train_on_batch(X_train, Y_train, learning_rate=0.1, epochs=20)
    plot_learning_curve(costs, display=True)
    print(compute_accuracy(model.predict(X_train), Y_train))

# Test a dense model
test_dense_model(X_train, Y_train)



def test_conv_model(X_train, Y_train):
    conv1 = Conv2DLayer(prev_layer=None, filters=3, kernel_size=5, strides=4, pad='same', activation='relu', input_shape=X_train.shape)
    flat1 = FlattenLayer(prev_layer=conv1)
    dense1 = DenseLayer(prev_layer=flat1, units=1, activation='sigmoid')
    dense2 = DenseLayer(prev_layer=dense1, units=10, activation='softmax')
    layers = [conv1, flat1, dense1, dense2]

    model = Model(layers)

    print('Model input shape: {}'.format(model.input_shape))
    #model.check_gradients(X_train, Y_train)
    
    costs = model.train_on_batch(X_train, Y_train, learning_rate=0.1, epochs=20)
    plot_learning_curve(costs, display=True)
    print(compute_accuracy(model.predict(X_train), Y_train))
    
# Test a convolutional model
test_conv_model(X_train, Y_train)


# Test conv with pool
def test_conv_with_pool_model(X_train, Y_train):
    conv1 = Conv2DLayer(prev_layer=None, filters=8, kernel_size=5, strides=4, pad='same', activation='relu', input_shape=X_train.shape)
    conv2 = Conv2DLayer(prev_layer=conv1, filters=16, kernel_size=5, strides=4, pad='same', activation='relu')
    pool1 = Pooling2DLayer(prev_layer=conv2, pool_size=2, strides=2, mode='max')
    conv3 = Conv2DLayer(prev_layer=pool1, filters=32, kernel_size=2, strides=1, pad='same', activation='relu')
    conv4 = Conv2DLayer(prev_layer=conv3, filters=64, kernel_size=2, strides=1, pad='same', activation='relu')
    flat1 = FlattenLayer(prev_layer=conv4)
    dense1 = DenseLayer(prev_layer=flat1, units=3, activation='sigmoid')
    dense2 = DenseLayer(prev_layer=dense1, units=10, activation='softmax')
    layers = [conv1, conv2, pool1, conv3, conv4, flat1, dense1, dense2]

    model = Model(layers)

    print('Model input shape: {}'.format(model.input_shape))
    
    #model.check_gradients(X_train, Y_train)
    costs = model.train_on_batch(X_train, Y_train, learning_rate=0.1, epochs=20)
    plot_learning_curve(costs, display=True)

    pred_classes = model.predict(X_train)
    
    acc = compute_accuracy(pred_classes, Y_train)
    print("The training accuracy is: {}".format(acc))


test_conv_with_pool_model(X_train, Y_train)

