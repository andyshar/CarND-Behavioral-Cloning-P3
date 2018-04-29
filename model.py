import csv
import cv2
import numpy as np

lines = []
with open('./data/driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        lines.append(line)

images = []
measurements = []
correction_factor = 0.2
keep_prob=0.5

for line in lines:
    for i in range(3):
        source_path = line[i]
        filename = source_path.split('\\')[-1]
        current_path = './data/IMG/' + filename
        image = cv2.imread(current_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        images.append(image)
        # center
        if i == 0:
            measurement = float(line[3])
        # left
        if i == 1:
            measurement = float(line[3]) + correction_factor
        # right
        if i == 2:
            measurement = float(line[3]) - correction_factor
        measurements.append(measurement)

## Flip the images
augmented_images = []
augmented_measurements = []

for image, measurement in zip(images, measurements):
    augmented_images.append(image)
    augmented_measurements.append(measurement)
    flipped_image = cv2.flip(image, 1)
    flipped_measurement = measurement * -1.0
    augmented_images.append(flipped_image)
    augmented_measurements.append(flipped_measurement)

X_train = np.array(augmented_images)
y_train = np.array(augmented_measurements)

print("images: " + str(len(augmented_images)))
print("measurements: " + str(len(augmented_measurements)))

from keras.models import Sequential
from keras.layers import Flatten, Dropout, Dense, Lambda, Activation, MaxPooling2D
from keras.layers.convolutional import Cropping2D, Convolution2D
from keras.optimizers import Adam

learning_rate = 1e-4
activation_relu = 'relu'

#model = Sequential()
#model.add(Flatten(input_shape=(160,320,3)))
#model.add(Dense(1))

#model = Sequential()
#model.add(Lambda(lambda x: x / 255.0 - 0.5, input_shape=(160,320,3)))
#model.add(Cropping2D(cropping=((70,25),(0,0))))
#model.add(Convolution2D(24, 5, 5, subsample=(2, 2), activation='relu'))
#model.add(Convolution2D(36, 5, 5, subsample=(2, 2), activation='relu'))
#model.add(Convolution2D(48, 5, 5, subsample=(2, 2), activation='relu'))
#model.add(Convolution2D(64, 3, 3, activation='relu'))
#model.add(Convolution2D(64, 3, 3, activation='relu'))
#model.add(Flatten())
#model.add(Dense(100))
#model.add(Dense(50))
#model.add(Dense(10))
#model.add((Dense(1))

model = Sequential()

model.add(Lambda(lambda x: x / 255.0 - 0.5, input_shape=(160,320,3)))
model.add(Cropping2D(cropping=((70,25),(0,0))))

model.add(Convolution2D(24, 5, 5, subsample=(2, 2), activation='relu'))
#model.add(Convolution2D(24, 5, 5, border_mode='same', subsample=(2, 2)))
#model.add(Activation(activation_relu))
##model.add(MaxPooling2D(pool_size=(2, 2), strides=(1, 1)))

model.add(Convolution2D(36, 5, 5, border_mode='same', subsample=(2, 2)))
model.add(Activation(activation_relu))
#model.add(MaxPooling2D(pool_size=(2, 2), strides=(1, 1)))

model.add(Convolution2D(48, 5, 5, border_mode='same', subsample=(2, 2)))
model.add(Activation(activation_relu))
#model.add(MaxPooling2D(pool_size=(2, 2), strides=(1, 1)))

model.add(Convolution2D(64, 3, 3, border_mode='same', subsample=(1, 1)))
model.add(Activation(activation_relu))
#model.add(MaxPooling2D(pool_size=(2, 2), strides=(1, 1)))

model.add(Convolution2D(64, 3, 3, border_mode='same', subsample=(1, 1)))
model.add(Activation(activation_relu))
#model.add(MaxPooling2D(pool_size=(2, 2), strides=(1, 1)))

model.add(Dropout(keep_prob))

model.add(Flatten())

model.add(Dense(100))
model.add(Activation(activation_relu))

model.add(Dense(50))
model.add(Activation(activation_relu))

model.add(Dense(10))
model.add(Activation(activation_relu))

model.add(Dense(1))

#model.compile(loss='mse', optimizer='adam')
#model.compile(optimizer='adam', loss='mse')
model.compile(optimizer=Adam(1e-4), loss="mse", )
#model.compile(optimizer='adam', loss='mse', metrics=['accuracy'])
#model.compile(optimizer=optimizers.Adam(lr=1e-04), loss='mean_squared_error')
#model.compile(optimizer=optimizer, loss = 'categorical_crossentropy',metrics=['accuracy'])
model.fit(X_train, y_train, validation_split=0.2, shuffle=True, nb_epoch=3)

model.save('model.h5')
exit()