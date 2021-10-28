#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 18 18:47:05 2020

@author: Vivek
"""
from tensorflow.keras.models import load_model

from PIL import Image
import numpy as np
import tensorflow as tf
from time import time


def load(filename):
    np_image = Image.open(filename)
    np_image = np_image.resize((256,256))
    np_image = np.asarray(np_image, dtype= np.float32)
    np_image = np_image / 255
    np_image = np_image.reshape(-1,256,256,3)
    return np_image


class retinopathy:
    def __init__(self, filename):
        self.filename = filename

    def predictionretinopathy(self):
        # load model
        #model = load_model('retina_weights.hdf5')
        # Change to bellow
        #tflite_model_path ='retina.tflite'
        tflite_model_path ='retina_opt.tflite'

        ######## For tflite ######
        # Load the TFLite model and allocate tensors.
        interpreter = tf.lite.Interpreter(model_path=tflite_model_path)
        interpreter.allocate_tensors()

        # Get input and output tensors.
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()

        # Test the model on input data.
        input_shape = input_details[0]['shape']
        ######## tflite end ######

        # summarize model
        # model.summary()
        imagename = self.filename
        test_image = load(imagename)

        ####For TFlite
        # If tflite model used
        input_data = test_image

        interpreter.set_tensor(input_details[0]['index'], input_data)

        time_before = time()
        interpreter.invoke()
        time_after = time()
        total_tflite_time = time_after - time_before
        # print("Total prediction time for tflite without opt model is: ", total_tflite_time)

        output_data_tflite = interpreter.get_tensor(output_details[0]['index'])
        # print("The tflite w/o opt prediction for this image is: ", output_data_tflite)


        # change result to interpreter for tflite

        #result = model.predict(test_image)
        # or #
        result = interpreter.get_tensor(output_details[0]['index'])
        result = np.argmax(result)

        if result == 0:
            prediction = 'Mild diabetic retinopathy observed'
            return [prediction]
        elif result == 1:
            prediction = 'Moderate diabetic retinopathy observed'
            return [prediction]
        elif result == 2:
            prediction = 'No diabetic retinopathy observed'
            return [prediction]
        elif result == 3:
            prediction = 'Proliferate diabetic retinopathy observed'
            return [prediction]
        else:
            prediction = 'Severe diabetic retinopathy observed'
            return [prediction]
