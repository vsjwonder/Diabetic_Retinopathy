import numpy as np
from keras.models import load_model
from keras.preprocessing import image
import numpy as np
from keras.preprocessing import image

model = load_model('model_own.h5')

from PIL import Image
import numpy as np
from skimage import transform
def load(filename):
    np_image = Image.open(filename)
    np_image = np.array(np_image).astype('float32') / 255
    np_image = transform.resize(np_image, (224, 224, 3))
    np_image = np.expand_dims(np_image, axis=0)
    return np_image
test_image = load(r'F:\AI\iNeuron\MyWork\ImageAnalysis\VSJ_Malaria_Detection\Infected_Cell.png')

result = model.predict(test_image)
result = np.argmax(result, axis=1)

#from keras.preprocessing.image import ImageDataGenerator
'''
train_datagen = ImageDataGenerator(rescale = 1./255,
                                   shear_range = 0.2,
                                   zoom_range = 0.2,
                                   horizontal_flip = True)
test_datagen = ImageDataGenerator(rescale = 1./255)
training_set = train_datagen.flow_from_directory(r'F:\AI\DeepLearn_iNeuron\ImageAnalysis1st\DCData\Parent\Training',
                                                 target_size = (64, 64),
                                                 batch_size = 32,
                                                 class_mode = 'binary')

test_set = test_datagen.flow_from_directory(r'F:\AI\DeepLearn_iNeuron\ImageAnalysis1st\DCData\Parent\Testing',
                                            target_size = (64, 64),
                                            batch_size = 32,
                                            class_mode = 'binary')
training_set.class_indices '''
if result > 0.5:
    prediction = 'Uninfected'
    print(prediction)
    #return [{"image": prediction}]
else:
    prediction = 'Infected'
    print(prediction)
    #return [{"image": prediction}]