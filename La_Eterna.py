import keras
import tensorflow as tf
from keras.preprocessing.image import ImageDataGenerator
# from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint
import matplotlib.pyplot as plt

print(keras.__version__)

# image_generator = ImageDataGenerator(rescale=1/255, validation_split=0.2)
# train_data_path = r"D:\AI_Project\Unlocked_Challenge_4-Main\data_cleaned\Train"
# validation_data_path = r""
def plotImages(images_arr):
    fig, axes = plt.subplots(1, 5, figsize=(20, 20))
    axes = axes.flatten()
    for img, ax in zip(images_arr, axes):
        ax.imshow(img)
    plt.tight_layout()
    plt.show()


image_datagen = ImageDataGenerator(rescale=1. / 255, validation_split=0.2,
                                      rotation_range=40,
                                      width_shift_range=0.2,
                                      height_shift_range=0.2,
                                      shear_range=0.2,
                                      zoom_range=0.2,
                                      horizontal_flip=True,
                                      fill_mode='nearest')

training_data = image_datagen.flow_from_directory(directory=r"D:\AI_Project\Unlocked_Challenge_4-Main\data_cleaned\Train",  # this is the target directory
                                                     target_size=(256, 256),  # all images will be resized to 150x150
                                                     batch_size=32,
                                                     subset="training",
                                                     class_mode='categorical')  # since we use binary_crossentropy loss, we need binary labels

training_data.class_indices

# this is the augmentation configuration we will use for validation:
# only rescaling
# valid_datagen = ImageDataGenerator(rescale=1. / 255)

# this is a similar generator, for validation data
valid_data = image_datagen.flow_from_directory(directory=r"D:\AI_Project\Unlocked_Challenge_4-Main\data_cleaned\Train",
                                               shuffle=True,
                                               target_size=(256, 256),
                                               subset="validation",
                                               batch_size=32,
                                               class_mode='categorical')

images = [training_data[0][0][0] for i in range(5)]
plotImages(images)

model_path = 'model/la_eterna.h5'
checkpoint = ModelCheckpoint(model_path, monitor='val_accuracy', verbose=1, save_best_only=True, mode='max')
callbacks_list = [checkpoint]
print('Checkpoint Ok Roger')
# Building cnn model
cnn_model = keras.models.Sequential([
    keras.layers.Conv2D(filters=32, kernel_size=(3,3), activation="relu",input_shape=(256, 256, 3)),

    keras.layers.Conv2D(filters=64, kernel_size=(3,3), activation="relu"),
    keras.layers.MaxPooling2D(pool_size=(2, 2)),
    keras.layers.Dropout(0.5),
    keras.layers.Conv2D(filters=64, kernel_size=(3,3), activation="relu"),
    keras.layers.MaxPooling2D(pool_size=(2, 2)),
    keras.layers.Dropout(0.5),
    keras.layers.Conv2D(filters=128, kernel_size=(3,3), activation="relu"),
    keras.layers.MaxPooling2D(pool_size=(2, 2)),
    keras.layers.Dropout(0.5),

    keras.layers.Flatten(),  # neural network building
    keras.layers.Dense(units=64, activation='relu'),  # input layers
    keras.layers.Dropout(0.25),
    keras.layers.Dense(units=128, activation='relu'),
    keras.layers.Dropout(0.25),
    keras.layers.Dense(units=256, activation='relu'),
    keras.layers.Dropout(0.25),
    keras.layers.Dense(units=2, activation='sigmoid')  # output layer
])
print('Summary')
cnn_model.summary()
# compile cnn model
cnn_model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0005), loss='categorical_crossentropy', metrics=['accuracy'])
print('Roger that')
print('training cnn model.-.-.-.-.')
history = cnn_model.fit(training_data,
                        epochs=50,
                        verbose=1,
                        validation_data=valid_data,
                        callbacks=callbacks_list)  # time start 16.06

# summarize history for accuracy
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
# summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

history.history

print('Roger that')
