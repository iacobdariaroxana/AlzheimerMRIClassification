import numpy
import tensorflow as tf
import numpy as np
import os

train_dataset = tf.keras.preprocessing.image_dataset_from_directory(
    '../prepdata/train', batch_size=64, image_size=(176, 208), label_mode='categorical')
validation_dataset = tf.keras.preprocessing.image_dataset_from_directory('../prepdata/val', batch_size=64,
                                                                         image_size=(176, 208),
                                                                         label_mode='categorical')
test_dataset = tf.keras.preprocessing.image_dataset_from_directory(
    '../prepdata/test', batch_size=64, image_size=(176, 208), label_mode='categorical')


x_train = np.concatenate([x for x, y in train_dataset][:-1])
# x_train = tf.expand_dims(x_train, axis=3, name=None)
# x_train = tf.repeat(x_train, 3, axis=3)
y_train = np.concatenate([y for x, y in train_dataset][:-1])

print(x_train.shape)


def inception(x,
              filters_1x1,
              filters_3x3_reduce,
              filters_3x3,
              filters_5x5_reduce,
              filters_5x5,
              filters_pool):
    path1 = tf.keras.layers.Conv2D(filters_1x1, (1, 1), padding='same', activation='relu')(x)

    path2 = tf.keras.layers.Conv2D(filters_3x3_reduce, (1, 1), padding='same', activation='relu')(x)
    path2 = tf.keras.layers.Conv2D(filters_3x3, (1, 1), padding='same', activation='relu')(path2)

    path3 = tf.keras.layers.Conv2D(filters_5x5_reduce, (1, 1), padding='same', activation='relu')(x)
    path3 = tf.keras.layers.Conv2D(filters_5x5, (1, 1), padding='same', activation='relu')(path3)

    path4 = tf.keras.layers.MaxPool2D((3, 3), strides=(1, 1), padding='same')(x)
    path4 = tf.keras.layers.Conv2D(filters_pool, (1, 1), padding='same', activation='relu')(path4)

    return tf.concat([path1, path2, path3, path4], axis=3)


inputs = tf.keras.Input(shape=(176, 208, 3))

x = tf.keras.layers.Conv2D(64, 7, strides=2, padding='same', activation='relu')(inputs)
x = tf.keras.layers.MaxPooling2D(3, strides=2)(x)

x = tf.keras.layers.Conv2D(64, 1, strides=1, padding='same', activation='relu')(x)
x = tf.keras.layers.Conv2D(192, 3, strides=1, padding='same', activation='relu')(x)

x = tf.keras.layers.MaxPooling2D(3, strides=2)(x)

x = inception(x,
              filters_1x1=64,
              filters_3x3_reduce=96,
              filters_3x3=128,
              filters_5x5_reduce=16,
              filters_5x5=32,
              filters_pool=32)

x = inception(x,
              filters_1x1=128,
              filters_3x3_reduce=128,
              filters_3x3=192,
              filters_5x5_reduce=32,
              filters_5x5=96,
              filters_pool=64)

x = tf.keras.layers.MaxPooling2D(3, strides=2)(x)

x = inception(x,
              filters_1x1=192,
              filters_3x3_reduce=96,
              filters_3x3=208,
              filters_5x5_reduce=16,
              filters_5x5=48,
              filters_pool=64)

aux1 = tf.keras.layers.AveragePooling2D((5, 5), strides=3)(x)
aux1 = tf.keras.layers.Conv2D(128, 1, padding='same', activation='relu')(aux1)
aux1 = tf.keras.layers.Flatten()(aux1)
aux1 = tf.keras.layers.Dense(1024, activation='relu')(aux1)
aux1 = tf.keras.layers.Dropout(0.7)(aux1)
aux1 = tf.keras.layers.Dense(4, activation='softmax')(aux1)

x = inception(x,
              filters_1x1=160,
              filters_3x3_reduce=112,
              filters_3x3=224,
              filters_5x5_reduce=24,
              filters_5x5=64,
              filters_pool=64)

x = inception(x,
              filters_1x1=128,
              filters_3x3_reduce=128,
              filters_3x3=256,
              filters_5x5_reduce=24,
              filters_5x5=64,
              filters_pool=64)

x = inception(x,
              filters_1x1=112,
              filters_3x3_reduce=144,
              filters_3x3=288,
              filters_5x5_reduce=32,
              filters_5x5=64,
              filters_pool=64)

aux2 = tf.keras.layers.AveragePooling2D((5, 5), strides=3)(x)
aux2 = tf.keras.layers.Conv2D(128, 1, padding='same', activation='relu')(aux2)
aux2 = tf.keras.layers.Flatten()(aux2)
aux2 = tf.keras.layers.Dense(1024, activation='relu')(aux2)
aux2 = tf.keras.layers.Dropout(0.7)(aux2)
aux2 = tf.keras.layers.Dense(4, activation='softmax')(aux2)

x = inception(x,
              filters_1x1=256,
              filters_3x3_reduce=160,
              filters_3x3=320,
              filters_5x5_reduce=32,
              filters_5x5=128,
              filters_pool=128)

x = tf.keras.layers.MaxPooling2D(3, strides=2)(x)

x = inception(x,
              filters_1x1=256,
              filters_3x3_reduce=160,
              filters_3x3=320,
              filters_5x5_reduce=32,
              filters_5x5=128,
              filters_pool=128)

x = inception(x,
              filters_1x1=384,
              filters_3x3_reduce=192,
              filters_3x3=384,
              filters_5x5_reduce=48,
              filters_5x5=128,
              filters_pool=128)

x = tf.keras.layers.GlobalAveragePooling2D()(x)

x = tf.keras.layers.Dropout(0.4)(x)
out = tf.keras.layers.Dense(4, activation='softmax')(x)

model = tf.keras.Model(inputs=inputs, outputs=[out, aux1, aux2])
model.summary()

filename = os.path.splitext(os.path.basename(__file__))[0]

callbacks = [
    tf.keras.callbacks.ModelCheckpoint(filepath=f'Model/{filename}-epoch_' + '{epoch:02d}', save_freq='epoch')
]

model.compile(optimizer='adam', loss=[tf.keras.losses.categorical_crossentropy,
                                      tf.keras.losses.categorical_crossentropy,
                                      tf.keras.losses.categorical_crossentropy],
              loss_weights=[1, 0.3, 0.3], metrics=['accuracy'])

history = model.fit(x_train, [y_train, y_train, y_train], batch_size=64, epochs=15)
print(history.history)
np.save(f'History/{filename}-history.npy', history.history)

loss, acc = model.evaluate(test_dataset)
print("loss: %.2f" % loss)
print("acc: %.2f" % acc)
