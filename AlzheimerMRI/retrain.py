import tensorflow as tf
import numpy as np

train_dataset = tf.keras.preprocessing.image_dataset_from_directory(
    'prepdata/train', batch_size=64, image_size=(176, 208), color_mode='grayscale')
validation_dataset = tf.keras.preprocessing.image_dataset_from_directory('prepdata/val', batch_size=64, image_size=(176,208), color_mode='grayscale')
test_dataset = tf.keras.preprocessing.image_dataset_from_directory(
    'prepdata/test', batch_size=64, image_size=(176, 208), color_mode='grayscale')
model = tf.keras.models.load_model("models/Model/model-6-epoch_05")

callbacks = [
    tf.keras.callbacks.ModelCheckpoint(filepath=f'models/Model/model-6(retrain)-epoch_' + '{epoch:02d}',
                                       save_freq='epoch')
]
history = model.fit(train_dataset, epochs=5, callbacks=callbacks, shuffle=True, validation_data=validation_dataset)

print(history.history)
np.save(f'models/History/model-6(retrain)-history.npy', history.history)

loss, acc = model.evaluate(test_dataset)
print("loss: %.2f" % loss)
print("acc: %.2f" % acc)
