from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt

# test_dataset = keras.preprocessing.image_dataset_from_directory(
#     'prepdata/test', batch_size=64, image_size=(176, 208), label_mode='categorical')
# 
# test_loss = []
# test_accuracy = []
# for i in range(1, 6):
#     model = keras.models.load_model("models/Model/model-1-epoch_{:0>2d}".format(i))
#     loss, acc = model.evaluate(test_dataset)
#     test_loss += [loss]
#     test_accuracy += [acc]
#     print("loss: %.2f" % loss)
#     print("acc: %.2f" % acc)
# 
# 
# # history loading
# history = np.load("models/History/model-1-history.npy", allow_pickle=True).item()
# print(history)
# 
# fig, axs = plt.subplots(2, 1, figsize=(15, 15))
# fig.tight_layout(pad=8)
# axs[0].plot(history['loss'])
# axs[0].plot(history['val_loss'])
# axs[0].plot(test_loss)
# axs[0].title.set_text('Training Loss vs Validation Loss vs Test Loss')
# axs[0].set_xlabel('Epochs')
# axs[0].set_ylabel('Loss')
# axs[0].legend(['Train', 'Val', 'Test'])
# 
# 
# axs[1].plot(history['accuracy'])
# axs[1].plot(history['val_accuracy'])
# axs[1].plot(test_accuracy)
# axs[1].title.set_text('Training Accuracy vs Validation Accuracy vs Test Accuracy')
# axs[1].set_xlabel('Epochs')
# axs[1].set_ylabel('Accuracy')
# axs[1].legend(['Train', 'Val', 'Test'])
# 
# plt.savefig('plots/model-1.png')
# plt.show()


# model1-epoch4 -> 0.87, model2-epoch4 -> 0.91, model3-epoch04 -> 0.89, model4-epoch5 -> 0.84,
# model5-epoch5 -> 0.89, model6-epoch5 -> 0.94, model7-epoch10 -> 0.96, model8-epoch09 -> 0.86
# model9-epoch5 -> 0.90, model10-epoch 7,8,9 -> 0.96


# model10, model7(0.96) -> model6 (0.94) -> model2 (0.91), model9(0.90), model3 model5(0.89), model1 (0.87), model8 (0.86), model4(0.84)

for i in range(1, 11):
    try:
        print("-------------------------------------------------------------")
        print("Model: ", i)
        model = keras.models.load_model(f"models/Model/model-{i}-epoch_01")
        model.summary()
        print("-------------------------------------------------------------")
    except:
        continue