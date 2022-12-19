from tensorflow import keras
import numpy as np

model = keras.models.load_model("models/Model/model-10-epoch_05")
model.summary()
test_dataset = keras.preprocessing.image_dataset_from_directory(
    'prepdata/test', batch_size=64, image_size=(176, 208))

loss, acc = model.evaluate(test_dataset)
print("loss: %.2f" % loss)
print("acc: %.2f" % acc)

# history loading
history = np.load("models/History/model-10-history.npy", allow_pickle=True).item()
print(history)

# model1-epoch4, model3-epoch5(0.82), model4-epoch4(0.83), model5-epoch5(0.86), model6-epoch5
# model1'-epoch5, model2'-epoch5(0.9686), model3'-epoch5(0.9790), model-4'-epoch4(0.9439)
# model-5'-epoch5(0.9757), model-6'-epoch5(0.9787)


# model1-epoch4 -> 0.87, model2-epoch4 -> 0.91, model3-epoch04 -> 0.89, model4-epoch5 -> 0.84,
# model5-epoch5 -> 0.89, model6-epoch5 -> 0.94, model7-epoch10 -> 0.96, model8-epoch09 -> 0.86
# model9-epoch5 -> 0.90, model10-epoch 7,8,9 -> 0.96
