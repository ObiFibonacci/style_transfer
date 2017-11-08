import numpy as np
from PIL import Image

from keras.applications.vgg16 import VGG16
from keras.applications.vgg16 import preprocess_input, decode_predictions

model = VGG16(weights='imagenet', include_top=True)

layers = dict([(layer.name, layer.output) for layer in model.layers])
layers

model.count_params()

image_path = '*/images/elephant.jpg'
image = Image.open(image_path)
image = image.resize((224,224))
image

x = np.array(image, dtype-'float32')
x = np.expand_dims(x, axis = 0)

x = preprocess_input(x)

preds = model.predict(x)
print('Predicted:', decode_predictions(pres, top=3)[0])
