
import tensorflow
# https://www.tensorflow.org/versions/r2.8/api_docs/python/tf/config/experimental/enable_op_determinism
tensorflow.keras.utils.set_random_seed(0)   
tensorflow.config.experimental.enable_op_determinism()




from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Input
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.layers import Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ReduceLROnPlateau


from . import logger

class LSTMModel:
    def __init__(self, input_shape, num_classes: int, load = False):
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.logger = logger
        if not load:
            self.model = self.build_model()
            self.check_repro()
        else:
            self.model = self.load("prova.h5")
        self.logging("LSTM model initialization complete.")

    def build_model(self):
        model = Sequential()
        model.add(Input(shape=self.input_shape))
        model.add(LSTM(64, return_sequences=False))
        model.add(Dense(32, activation='relu'))
        model.add(Dense(16, activation='relu'))
        model.add(Dense(self.num_classes, activation='softmax'))
        optimizer = Adam(learning_rate=0.001)
        model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['categorical_accuracy'])
        self.logging("The model is compiled.")
        return model

    def summary(self):
        self.model.summary()

    def fit(self, x_train, y_train, epochs=100, batch_size=16, validation_data=None):
        early_stopping = EarlyStopping(
            monitor='val_loss', 
            patience=10, 
            restore_best_weights=True
        )
        
        # Define the learning rate scheduler
        lr_scheduler = ReduceLROnPlateau(
            monitor='val_loss',        # Metric to monitor (can also be 'val_categorical_accuracy')
            factor=0.5,                # Factor by which the learning rate will be reduced
            patience=2,                # Number of epochs with no improvement before reducing the learning rate
            min_lr=1e-6,               # Minimum learning rate
            verbose=1                  # Print updates to the console
)
        
        self.history = self.model.fit(x_train, y_train, epochs=epochs, batch_size=batch_size, validation_data=validation_data, 
                                callbacks=[lr_scheduler,early_stopping])
        return self.history

    def predict(self, x):
        return self.model.predict(x)
    
    def logging(self, sentence):
        logger.info(sentence)
        
    def save(self,filename):
        self.model.model.save(filename + '.h5')
        return 
    
    def load(self,path):
        model = Sequential()
        model.add(Input(shape=self.input_shape))
        model.add(LSTM(64, return_sequences=False))
        model.add(Dense(32, activation='relu'))
        model.add(Dense(16, activation='relu'))
        model.add(Dense(self.num_classes, activation='softmax'))
        model.load_weights(path)
        return model
    
    
    def check_repro(self):
        labels = tensorflow.random.normal((1, 10000))
        logits = tensorflow.random.normal((1, 10000))
        output = tensorflow.nn.softmax_cross_entropy_with_logits(labels=labels,
                                                        logits=logits)
        self.logging("checking for reproducibility!")
        self.logging("the value is {}".format(output))
        for i in range(5):
            output2 = tensorflow.nn.softmax_cross_entropy_with_logits(labels=labels,
                                                                logits=logits)
            tensorflow.debugging.assert_equal(output, output2)
            self.logging("check #{}, the result is {}".format(i,output2))
            
        self.logging("check for reproducibility done!")
