import time
import logging


class ArkML(object):
    def __init__(self):
        # Load up the logging mechanism
        self.log = self.spawnLogger(self.__class__.__name__)
        self.log.debug("Instantiated the " + self.__class__.__name__ + " class.")

    def spawnLogger(self, className):
        """
        Simple function to setup the logging mechanism.
        """
        tzTime = time.strftime("%z", time.gmtime())
        tzName = time.strftime("%Z", time.gmtime())

        # Create the logger object;
        logger = logging.getLogger(className)
        logger.setLevel(logging.DEBUG)

        # Create the file handler which logs even debug messages;
        fh = logging.FileHandler('arkml.log')
        fh.setLevel(logging.DEBUG)

        # Create the console handler with a higher log level
        ch = logging.StreamHandler()
        # ch.setLevel(logging.INFO)
        ch.setLevel(logging.INFO)

        # create formatter and add it to the handlers
        # formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s', '%m/%d/%Y %I:%M:%S %p')
        fh_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                                         '%Y-%m-%d %H:%M:%S' + tzTime)
        ch_formatter = logging.Formatter('[%(asctime)s] %(levelname)s\t%(message)s', '%H:%M:%S' + tzTime)
        fh.setFormatter(fh_formatter)
        ch.setFormatter(ch_formatter)
        # add the handlers to the logger
        logger.addHandler(fh)
        logger.addHandler(ch)

        return logger

    def MNIST_helloWorld(self):
        from keras.datasets import mnist
        from keras import models, layers
        from keras.utils import to_categorical

        (train_images, train_labels), (test_images, test_labels) = mnist.load_data()

        # Let's look at the training data:
        header = "[train_images] "
        self.log.info(header + "Shape:\t" + str(train_images.shape))
        self.log.info(header + "Length:\t" + str(len(train_images)))
        self.log.info(header + "Content:\t" + str(train_labels))

        # And the test data
        header = "[test_images] "
        self.log.info(header + "Shape:\t" + str(test_images.shape))
        self.log.info(header + "Length:\t" + str(len(test_images)))
        self.log.info(header + "Content:\t" + str(test_labels))

        # Let's build the network
        network = models.Sequential()

        # The core building block of neural networks is the layer, a data-processing module that
        # you can think of as a filter for data. Some data goes in, and it comes out in a more useful form.

        # Our network consists of a sequence of two Dense layers, which are densely connected
        # (also called fully connected) neural layers.
        network.add(layers.Dense(512, activation='relu', input_shape=(28 * 28,)))
        # The second (and last) layer is a 10-way softmax layer,
        # which means it will return an array of 10 probability scores (summing to 1).
        # Each score will be the probability that the current digit image belongs to one of our 10 digit classes.
        network.add(layers.Dense(10, activation='softmax'))

        # To make the network ready for training, we need to pick three more things, as part of the compilation step:

        ### A loss function
        # How the network will be able to measure its performance on the training data,
        # and thus how it will be able to steer itself in the right direction.

        ### An optimizer
        # The mechanism through which the network will update itself based on the data it sees and its loss function.

        ### Metrics to monitor during training and testing
        # Here, we’ll only care about accuracy.
        # (the fraction of the images that were correctly classified)
        network.compile(optimizer='rmsprop',
                        loss='categorical_crossentropy',
                        metrics=['accuracy'])

        # Before training, we’ll preprocess the data by reshaping it into the shape the network expects
        # and scaling it so that all values are in the [0, 1] interval.
        # Previously, our training images, for instance, were stored in an array of shape (60000, 28, 28)
        # of type uint8 with values in the [0, 255] interval.
        # We transform it into a float32 array of shape (60000, 28 * 28) with values between 0 and 1.
        train_images = train_images.reshape((60000, 28 * 28))
        train_images = train_images.astype('float32') / 255

        test_images = test_images.reshape((10000, 28 * 28))
        test_images = test_images.astype('float32') / 255

        # We also need to categorically encode the labels; more on that later...
        train_labels = to_categorical(train_labels)
        test_labels = to_categorical(test_labels)

        # Train the network
        # In Keras this is done via a call to the network’s fit method
        # (we fit the model to its training data)
        network.fit(train_images, train_labels, epochs=5, batch_size=128)

        test_loss, test_acc = network.evaluate(test_images, test_labels)
        time.sleep(2)  # wait for evaluate() output
        self.log.info('[RESULT] Test accuracy:\t' + "{0:.2f}%".format(test_acc * 100))
        self.log.info('[RESULT] Test loss:\t' + "{0:.2f}%".format(test_loss * 100))

    def naive_relu(self, x, BLAS=False):
        """
        The "relu" operation and addition are element-wise operations: operations that are applied independently
        to each entry in the tensors being considered.

        This means these operations are highly amenable to massively parallel implementations
        (vectorized implementations, a term that comes from the vector processor supercomputer architecture
        from the 1970–1990 period).

        If you want to write a naive Python implementation of an element-wise operation,
        you use a for loop, as in this naive implementation of an element-wise "relu" operation.

        It's of course better to use the well-optimized built-in Numpy functions,
        which themselves delegate the heavy lifting to a Basic Linear Algebra Subprograms (BLAS) implementation.

        BLAS are low-level, highly parallel, efficient tensor-manipulation routines that are typically implemented in Fortran or C.

        :param x: a Numpy 2D array
        :return: a Numpy 2D array
        """

        from numpy import ndarray

        # Check if x is a 2D Numpy tensor
        assert isinstance(x, ndarray)
        assert len(x.shape) == 2

        if not BLAS:
            # Avoid overwriting the input tensor
            x = x.copy()
            for i in range(x.shape[0]):
                for j in range(x.shape[1]):
                    x[i, j] = max(x[i, j], 0)

        else:
            from numpy import maximum
            x = maximum(x, 0.)

        return x

    def naive_add(self, x, y):
        """
        See self.naive_add()

        This is the example of naive addition.
        On the same principle, you can do element-wise multiplication, subtraction, and so on.

        :param x: a Numpy 2D array
        :param y: a Numpy 2D array
        :return: a Numpy 2D array
        """
        from numpy import ndarray

        # Check if x is a 2D Numpy tensor
        assert isinstance(x, ndarray)
        assert isinstance(y, ndarray)
        assert len(x.shape) == 2
        assert x.shape == y.shape

        x = x.copy()
        for i in range(x.shape[0]):
            for j in range(x.shape[1]):
                x[i, j] += y[i, j]

        return x

    def binary_classification(self):
        """
        Just like the MNIST dataset, the IMDB dataset comes packaged with Keras.
        It has already been preprocessed: the reviews (sequences of words) have been turned
        into sequences of integers, where each integer stands for a specific word in a dictionary.

        :return:
        """

        import matplotlib.pyplot as plt
        import numpy as np
        from keras.datasets import imdb
        from keras import models, layers, optimizers, losses, metrics


        (train_data, train_labels), (test_data, test_labels) = imdb.load_data(num_words=10000)

        # If you're curious in decoding the reviwes back to English:
        word_index = imdb.get_word_index()
        reverse_word_index = dict([(value, key) for (key, value) in word_index.items()])
        decoded_review = ' '.join([reverse_word_index.get(i - 3, '?') for i in train_data[0]])

        # Function that creates an all-zero matrix of shape (len(sequences), dimension)
        def vectorize_sequences(sequences, dimension=10000):
            results = np.zeros((len(sequences), dimension))
            for i, sequence in enumerate(sequences):
                # Set specific indices of results[i] to 1s
                results[i, sequence] = 1.
            return results

        # Vectorized training and test data
        x_train = vectorize_sequences(train_data)
        x_test = vectorize_sequences(test_data)

        # Also vectorize the labels (these are made up 0s and 1s; positive or negative review)
        y_train = np.asarray(train_labels).astype('float32')
        y_test = np.asarray(test_labels).astype('float32')

        # Define the model
        model = models.Sequential()
        model.add(layers.Dense(16, activation='relu', input_shape=(10000,)))
        model.add(layers.Dense(16, activation='relu'))
        model.add(layers.Dense(1, activation='sigmoid'))

        # We can customize the optimizer, the losses and the metrics
        # model.compile(optimizer='rmsprop',
        model.compile(optimizer=optimizers.RMSprop(lr=0.001),
                      loss=losses.binary_crossentropy,
                      metrics=[metrics.binary_accuracy])

        # In order to monitor during training the accuracy of the model on data it has never seen before,
        # you’ll create a validation set by setting apart 10,000 samples from the original training data.
        x_val = x_train[:10000]
        partial_x_train = x_train[10000:]
        y_val = y_train[:10000]
        partial_y_train = y_train[10000:]

        history = model.fit(partial_x_train,
                            partial_y_train,
                            epochs=20,
                            batch_size=512,
                            validation_data=(x_val, y_val))

        history_dict = history.history
        history_dict.keys()


        history_dict = history.history
        loss_values = history_dict['loss']
        val_loss_values = history_dict['val_loss']
        epochs = range(1, len(history_dict['binary_accuracy']) + 1)

        plt.plot(epochs, loss_values, 'bo', label='Training loss')
        plt.plot(epochs, val_loss_values, 'b', label='Validation loss')
        plt.title('Training and validation loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()
        plt.show()

        plt.clf()
        acc_values = history_dict['binary_accuracy']
        val_acc_values = history_dict['val_binary_accuracy']
        plt.plot(epochs, acc_values, 'bo', label='Training acc')
        plt.plot(epochs, val_acc_values, 'b', label='Validation acc')
        plt.title('Training and validation accuracy')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()
        plt.show()

        print(model.predict(x_test))

    def multiclass_classification(self):
        import numpy as np
        import matplotlib.pyplot as plt
        from keras.datasets import reuters
        from keras import models, layers, optimizers, losses, metrics

        ############################################################
        (train_data, train_labels), (test_data, test_labels) = reuters.load_data(num_words=40000)

        ############################################################
        # Example of how to decode it back to words
        word_index = reuters.get_word_index()
        reverse_word_index = dict([(value, key) for (key, value) in word_index.items()])
        decoded_newswire = ' '.join([reverse_word_index.get(i - 3, '?') for i in train_data[0]])
        decoded_newswire_label = ' '.join([reverse_word_index.get(i - 3, '?') for i in train_labels])

        ############################################################
        # Vectorize the newswire phrases
        def vectorize_sequences(sequences, dimension=40000):
            results = np.zeros((len(sequences), dimension))

            for i, sequence in enumerate(sequences):
                results[i, sequence] = 1.

            return results

        x_train = vectorize_sequences(train_data)
        x_test = vectorize_sequences(test_data)

        ############################################################
        # # Vectorize the labels
        # def to_one_hot(labels, dimension=46):
        #     results = np.zeros((len(labels), dimension))
        #
        #     for i, label in enumerate(labels):
        #         results[i, label] = 1.
        #
        #     return results
        #
        # one_hot_train_labels = to_one_hot(train_labels)
        # one_hot_test_labels = to_one_hot(test_labels)

        # Or you can use the built-in method :)
        from keras.utils.np_utils import to_categorical
        one_hot_train_labels = to_categorical(train_labels)
        one_hot_test_labels = to_categorical(test_labels)
        # sparse_categorical_crossentropy
        y_train = np.array(train_labels)
        y_test = np.array(test_labels)

        ############################################################
        model = models.Sequential()
        model.add(layers.Dense(64, activation='relu', input_shape=(40000,)))
        model.add(layers.Dense(512, activation='relu'))
        model.add(layers.Dense(46, activation='softmax'))

        ############################################################
        # model.compile(optimizer=optimizers.RMSprop(lr=0.001),
        model.compile(optimizer='rmsprop',
                      # loss=losses.categorical_crossentropy,
                      loss='sparse_categorical_crossentropy',
                      metrics=['acc'])

        ############################################################
        # Set apart some validation data
        x_val = x_train[:4000]
        y_val = one_hot_train_labels[:4000]

        partial_x_train = x_train[4000:]
        partial_y_train = one_hot_train_labels[4000:]

        # sparse_categorical_crossentropy
        y_val = y_train[:4000]
        partial_y_train = y_train[4000:]



        ############################################################
        # Train the data
        history = model.fit(partial_x_train,
                            partial_y_train,
                            epochs=7,
                            batch_size=512,
                            validation_data=(x_val, y_val))

        ############################################################
        loss = history.history['loss']
        val_loss = history.history['val_loss']
        epochs = range(1, len(loss) + 1)
        plt.plot(epochs, loss, 'r', label='Training loss')
        plt.plot(epochs, val_loss, 'b', label='Validation loss')
        plt.title('Training and validation loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()
        plt.show()
        plt.clf()
        acc = history.history['acc']
        val_acc = history.history['val_acc']
        plt.plot(epochs, acc, 'r', label='Training acc')
        plt.plot(epochs, val_acc, 'b', label='Validation acc')
        plt.title('Training and validation accuracy')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()
        plt.show()

        ############################################################
        # results = model.evaluate(x_test, one_hot_test_labels)
        results = model.evaluate(x_test, y_test)
        self.log.info('[RESULT] Test accuracy:\t' + "{0:.2f}%".format(results[1] * 100))
        # self.log.info('[RESULT] Test loss:\t' + "{0:.2f}%".format(results[0] * 100))

        ############################################################
        predictions = model.predict(x_test)
        pass
        # y_train = np.array(train_labels)
        # y_test = np.array(test_labels)

if __name__ == "__main__":
    ArkML = ArkML()

    # ArkML.MNIST_helloWorld()
    # ArkML.binary_classification()
    ArkML.multiclass_classification()