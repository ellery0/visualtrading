import keras.backend as K
import pickle
from keras import models
import glob
import numpy as np
from PIL import Image
from keras.models import load_model
import matplotlib.pyplot as plt

#model = pickle.load(open(('model.h5'),'rb'))
model = load_model('model.h5')

fname = '/Users/fuyuting/PycharmProjects/visualtrading/train-pic-7/traincandle_3.png'
train_image = np.array(Image.open(fname).resize((int(640/4),int(480/4)),Image.ANTIALIAS).convert('RGB'))
train_image = train_image.astype('float32') / 255
img_tensor = train_image
img_tensor = np.expand_dims(img_tensor, axis=0)


# # visualize layer outputs
#
# layer_outputs = [layer.output for layer in model.layers[:8]]
# activation_model = models.Model(inputs=model.input, outputs=layer_outputs)
# activations = activation_model.predict(img_tensor)
# first_layer_activation = activations[0]
#

# #Visualizing the fourth channel
# plt.matshow(first_layer_activation[0, :, :, 4], cmap='viridis')

# visualize filters

# visualize heatmap

img = train_image
x = img
x = np.expand_dims(x, axis=0)
candle_output = model.output[:, 1]

layer_names = [layer.name for layer in model.layers]

last_conv_layer = model.get_layer(layer_names[-4])
grads = K.gradients(candle_output, last_conv_layer.output)[0]
#print(grads.shape)
pooled_grads = K.mean(grads, axis=(0, 1, 2))
iterate = K.function([model.input],
                     [pooled_grads, last_conv_layer.output[0]])
pooled_grads_value, conv_layer_output_value = iterate([x])
for i in range(pooled_grads.shape[0]):
    conv_layer_output_value[:, :, i] *= pooled_grads_value[i]
heatmap = np.mean(conv_layer_output_value, axis=-1)
heatmap = np.maximum(heatmap, 0)
heatmap /= np.max(heatmap)
#print(heatmap)
plt.matshow(heatmap)


# Superimposing the heatmap with the original picture
import cv2

img = cv2.imread(fname)
heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
heatmap = np.uint8(255 * heatmap)
heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
superimposed_img = heatmap * 0.4 + img
#plt.matshow(superimposed_img)
cv2.imwrite('/Users/fuyuting/Downloads/candle13.png', superimposed_img)


























