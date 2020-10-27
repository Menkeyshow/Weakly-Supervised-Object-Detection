import tensorflow as tf

from tensorflow.keras.layers import Input, Conv2D, BatchNormalization, concatenate, UpSampling2D, MaxPooling2D, Reshape, Lambda, Activation
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam

import numpy as np

## Base neural network model ##
class base_model(object):
            def __init__(self, inputshape=(150, 200, 3)):
                self.inputshape = inputshape

            def ResizeLayer(self, inputs, dim):
	            return Lambda(lambda image: tf.image.resize(image, dim))(inputs)

            # Build model
            def getModel(self, _optimizerChoice="Adam"):
                conf1 = dict()
                conf1["activation"] = "relu"
                conf1["padding"] = "same"


                inputlayer = Input(shape=(150, 200, 3))

                x = Conv2D(16, 3, **conf1)(inputlayer)
                layer1 = BatchNormalization()(x)
                layer2 = MaxPooling2D(padding="same")(layer1)

                x = Conv2D(32, 3, **conf1)(layer2)
                x = BatchNormalization()(x)

                x = Conv2D(32, 3, **conf1)(x)
                x = BatchNormalization()(x)

                layer3 = concatenate([x, layer2], axis=3)
                layer4 = MaxPooling2D(padding="same")(layer3)

                x = Conv2D(64, 3, **conf1)(layer4)
                x = BatchNormalization()(x)

                x = Conv2D(64, 3, **conf1)(x)
                x = BatchNormalization()(x)

                x = concatenate([x, layer4], axis=3)

                x = Conv2D(128, 3, **conf1)(x)
                x = BatchNormalization()(x)

                x = Conv2D(128, 3, **conf1)(x)
                x = BatchNormalization()(x)

                x = UpSampling2D(size=(2, 2), data_format=None)(x)
                x = self.ResizeLayer(x, (75, 100))

                x = concatenate([x, layer3], axis=3)
                x = Conv2D(64, 3, **conf1)(x)
                x = BatchNormalization()(x)

                x = Conv2D(32, 3, **conf1)(x)
                x = BatchNormalization()(x)

                x = Conv2D(32, 3, **conf1)(x)
                x = BatchNormalization()(x)
                x = UpSampling2D(size=(2, 2), data_format=None)(x)
                x = self.ResizeLayer(x, (150, 200))

                x = concatenate([x, layer1], axis=3)

                x = Conv2D(16, 3, **conf1)(x)
                x = BatchNormalization()(x)

                x = Conv2D(16, 3, **conf1)(x)
                x = BatchNormalization()(x)

                x = Conv2D(1, 3, padding="same", activation="sigmoid")(x)

                predictions = Reshape((150, 200, 1))(x)
                #optimizerChoice = "%s()" % _optimizerChoice
                #print(optimizerChoice)
                model = Model(inputs=inputlayer, outputs=predictions)
                model.compile(optimizer=Adam(),
                    loss='binary_crossentropy',
                    metrics=['accuracy'])
                return model

# Train on dummy data
if __name__ == "__main__":
    number_train_images = 100
    number_test_images = 100
    number_epochs = 1

    # create dummy training data
    X_train = np.random.rand(number_train_images, 150, 200, 3) # images with rgb data
    Y_train = np.random.rand(number_train_images, 150, 200, 1) # number images x height x width

    #Validierungsdatengenerierung (random Bilder)
    X_test = np.random.rand(number_test_images, 150, 200, 3) # images with rgb data
    Y_test = np.random.rand(number_test_images, 150, 200, 1) # number images x height x width

    model_object = base_model()
    model = model_object.getModel()
    # plot_model(model, to_file='basismodel.png', show_shapes=True)
    # train model with dummy data X
    model.fit(X_train, Y_train, batch_size=1, epochs=number_epochs, verbose=1)

    # evaluate model with dummy data Y ==> Tupel aus Loss und Accuracy tuple (Loss, Accuracy)
    score = model.evaluate(X_test, Y_test, verbose=1)
