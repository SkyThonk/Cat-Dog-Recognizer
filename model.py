from keras.models import Sequential
from keras.layers import Convolution2D, MaxPooling2D, Flatten, Dense
import numpy as np
from keras.preprocessing import image

###It is for to remove some tensorflow warnings###
try:
    from tensorflow.python.util import module_wrapper as deprecation
except ImportError:
    from tensorflow.python.util import deprecation_wrapper as deprecation
deprecation._PER_MODULE_WARNING_LIMIT = 0

###Ignore the warnings tensorflow just give suggestions to use there module###
def ImagePrediction(loc):
    test_image = image.load_img(loc, target_size = (64,64))
    test_image = image.img_to_array(test_image)
    test_image = np.expand_dims(test_image, axis =0)
    return test_image

def main():
    ###Initialize the CNN###
    classifier = Sequential()

    ###Add Convolution Layer [Increase the "input_shape" "Ex:input_shape = (256,256,3)" to get better result but that require more time to train your model]###
    classifier.add(Convolution2D(32,3,3, input_shape = (64,64,3), activation = 'relu'))

    ###Add Max Pooling Layer###
    classifier.add(MaxPooling2D(pool_size=(2,2)))

    ###Add another convolution and Max Pooling layer to improve thr perfomance of your model###
    classifier.add(Convolution2D(32,3,3, activation = 'relu'))
    classifier.add(MaxPooling2D(pool_size=(2,2)))

    ###Add Flattening Layer###
    classifier.add(Flatten())

    ###Add the Neural Network Layer [Increase the "output_dim" "Ex:output_dim = 256" to get better result but that require more time to train your model]###
    classifier.add(Dense(output_dim=128,activation='relu'))
    ###Neural Network Output layer
    classifier.add(Dense(output_dim=1,activation= 'sigmoid'))

    ###Compile the Convolution Neural Network###
    classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

    ###Fitting the Convolution Neural Network to dataset###
    from keras.preprocessing.image import ImageDataGenerator

    train_datagen = ImageDataGenerator(rescale=1./255,shear_range = 0.2, zoom_range = 0.2, horizontal_flip= True)
    test_datagen = ImageDataGenerator(rescale = 1./255)

    training_set = train_datagen.flow_from_directory('dataset/training_set', target_size=(64,64),batch_size =32, class_mode = 'binary')
    test_set = test_datagen.flow_from_directory('dataset/test_set', target_size=(64,64),batch_size =32, class_mode = 'binary')

    ###Train Your Model [increase the nb_epoch value to increase the model accuracy but it will take much more time to train your model]###
    classifier.fit_generator(training_set, samples_per_epoch = 12000, nb_epoch = 25, validation_data = test_set, nb_val_samples = 2000)


    ###predict the images###
    img = input("Enter the location of your image file: ")
    ###If you don't have any image than you can enter this location 'dataset/single_prediction/cat_or_dog_2.jpg'###
    test_image = ImagePrediction(img)
    result = classifier.predict(test_image)
    if result[0][0] == 1:
        predictions = 'dog'
    else:
        predictions = 'cat'
    print('\n\n\nPrediction: ',predictions)

if __name__ == '__main__' :
    main()