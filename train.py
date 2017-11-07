import keras
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras import backend as K
from loading import load_dataset
from keras.models import load_model


batch_size = 128
epochs = 2
X_train, X_test, y_train, y_test,num_classes = load_dataset()
print(num_classes,'classes')

num_rows,num_cols = X_train[0].shape
input_shape = (num_rows, num_cols,1)

X_train = X_train.reshape(X_train.shape[0],num_rows, num_cols,1).astype('float32')/255.0
X_test = X_test.reshape(X_test.shape[0],num_rows, num_cols,1).astype('float32')/255.0

print(X_train.shape)
print(y_train.shape)

model = Sequential()
model.add(Conv2D(32, kernel_size=(3, 3),activation='relu',input_shape=input_shape))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Conv2D(32, kernel_size=(3, 3),activation='relu',input_shape=input_shape))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(256, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(num_classes, activation='softmax'))

model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.Adadelta(),
              metrics=['accuracy'])

model.save('models/m1.h5')
model = load_model('models/m1.h5')

model.fit(X_train, y_train,
          batch_size=batch_size,
          epochs=epochs,
          verbose=1,
          validation_data=(X_test, y_test))

model.save('models/m1.h5')
model = load_model('models/m1.h5')

score = model.evaluate(X_test, y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])