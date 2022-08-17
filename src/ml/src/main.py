import numpy as np
from tensorflow import keras
# import tensorflow
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import mnist


# importing the libraries
import tensorflow.keras.layers as layers
from tensorflow.keras.models import Model
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPool2D, Dense, Flatten


#defining function for building the model
def create_model(input_shape = (28,28,1)):
    model = keras.Sequential([
    layers.Conv2D(filters = 32, kernel_size = 3, activation = 'relu', padding = 'same', input_shape = input_shape),
    layers.MaxPool2D(pool_size = 2),
    
    layers.Conv2D(filters = 64, kernel_size = 3, activation = 'relu', padding = 'same'),
    layers.MaxPool2D(pool_size = 2),
    
    layers.Conv2D(filters = 128, kernel_size = 3, activation = 'relu', padding = 'same'),
    layers.MaxPool2D(pool_size = 2),
    
    layers.Flatten(),
    layers.Dense(units = 54, activation = 'relu'),
    layers.Dense(units = 10, activation = 'softmax')])
    
    return model

def compile_model(model, optimizer='adam', loss='categorical_crossentropy'):
    model.compile(optimizer=optimizer, loss=loss, metrics=["accuracy"])

def fitting_model(model, x, y, epoch):
    model.fit(x,y, shuffle = True, epochs = epoch)

resize_image_length = 120
class_size = 5

def main():
    #loading dataset
    (train_X, train_y), (val_X, val_y) = mnist.load_data()
    print("trainx", train_X[0], train_X.shape)
    print("trainy", train_y[0], train_y.shape)

    print("valx", val_X[0], val_X.shape)
    print("valy", val_y[0], val_y.shape)

    # exit()
    #normalizing the dataset
    train_X, val_X = train_X/255, val_X/255

    # visualizing 9 rndom digits from the dataset
    for i in range(331,340):
        plt.subplot(i)
        a = np.random.randint(0, train_X.shape[0], 1)
        plt.imshow(train_X[a[0]], cmap = plt.get_cmap('binary'))

    plt.tight_layout()
    plt.show()

    #reshaping the independant variables
    train_X = train_X.reshape(train_X.shape[0], resize_image_length, resize_image_length, 1)
    val_X = val_X .reshape(val_X.shape[0], resize_image_length, resize_image_length, 1)

    #encoding the dependant variable
    train_y = np.eye(class_size)[train_y]
    val_y = np.eye(class_size)[val_y]

    #creating model
    model = create_model((resize_image_length, resize_image_length,1))
    #optimizing model
    compile_model(model, 'adam', 'categorical_crossentropy')

    #training model
    history = model.fit(train_X, train_y, validation_data = (val_X, val_y), batch_size = 150, epochs = 8)
    model.save("cnn_digitclass.model") #model will be save in root folder to be later called out for prediction

    #model performance visualization
    f = plt.figure(figsize=(20,8))

    #accuracy
    plt1 = f.add_subplot(121)
    plt1.plot(history.history['accuracy'], label = str('Training accuracy'))
    plt1.plot(history.history['val_accuracy'], label = str('Validation accuracy'))
    plt.legend()
    plt.title('accuracy')

    #loss
    plt2 = f.add_subplot(122)
    plt2.plot(history.history['loss'], label = str('Training loss'))
    plt2.plot(history.history['val_loss'], label = str('Validation loss'))
    plt.legend()
    plt.title('loss')

    plt.show()

if __name__ == '__main__':
    main()