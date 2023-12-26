import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.applications import VGG16
from tensorflow.keras import models
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.vgg16 import preprocess_input


model = VGG16(weights='imagenet', include_top=False)

for layer in model.layers:
    print(layer.name)

# initial_layer_names = ['block1_conv1', 'block1_conv2'] 
# initial_layer_outputs = [model.get_layer(name).output for name in initial_layer_names]


layer_name = 'block1_conv1'
selected_layer = model.get_layer(layer_name)
selected_layer_output = selected_layer.output

activation_model = models.Model(inputs=model.input, outputs=selected_layer_output)

def visualize_filters(layer_name, num_filters=10, figsize=(10, 10)):
    filters = model.get_layer(layer_name).get_weights()[0]
    fig, ax = plt.subplots(nrows=num_filters // 4, ncols=4, figsize=figsize)
    for i in range(num_filters):
        ax[i // 4, i % 4].imshow(filters[:, :, 0, i], cmap='viridis')
        ax[i // 4, i % 4].axis('off')
    plt.tight_layout()
    plt.show()


def visualize_feature_maps(img_path):
    img = image.load_img(img_path, target_size=(224, 224))
    img_tensor = image.img_to_array(img)
    img_tensor = np.expand_dims(img_tensor, axis=0)
    img_tensor = preprocess_input(img_tensor)

    activation = activation_model.predict(img_tensor)

    num_filters = activation.shape[-1]
    rows = (num_filters // 8) + 1
    fig, ax = plt.subplots(rows, 8, figsize=(16, rows * 2))
    for i in range(num_filters):
        ax[i // 8, i % 8].imshow(activation[0, :, :, i], cmap='viridis')
        ax[i // 8, i % 8].axis('off')
    plt.tight_layout()
    plt.show()


# def visualize_feature_maps(img_path):
#     img = image.load_img(img_path, target_size=(224, 224))
#     img_tensor = image.img_to_array(img)
#     img_tensor = np.expand_dims(img_tensor, axis=0)
#     img_tensor = preprocess_input(img_tensor)

#     activations = activation_model.predict(img_tensor)

#     for i, activation in enumerate(activations):
#         num_filters = activation.shape[-1]
#         rows = (num_filters // 8) + 1
#         fig, ax = plt.subplots(rows, 8, figsize=(16, rows * 2))
#         for j in range(num_filters):
#             ax[j // 8, j % 8].imshow(activation[0, :, :, j], cmap='viridis')
#             ax[j // 8, j % 8].axis('off')
#         plt.tight_layout()
#         plt.show()

visualize_filters(layer_name)

image_path = 'Images/pexels-public-domain-pictures-41315.jpg'  # Replace with the path to your image
visualize_feature_maps(image_path)

