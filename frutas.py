import keras
from keras.models import load_model
import time
import numpy as np
from PIL import Image
from keras.preprocessing import image
import tensorflow as tf
print(tf.VERSION)
print(keras.__version__)
print("keras version should be 2.2.4")


class Mlmodel:


	team ="ML Avengers"
	name = "fruits3"
	global path_modelo
	path_modelo = "fruits3.h5"
	path_imagen = ""

	def predict(path_imagen,path_modelo):

		model = load_model(path_modelo)

		tic=time.time()



		img=image.load_img(path_imagen, target_size=(150,150))
		img_tensor=image.img_to_array(img)
		img_tensor=np.expand_dims(img_tensor, axis=0)
		img_tensor/=255
		results = model.predict(img_tensor)


		#print(results)
		time_execution = time.time()-tic
		time_execution = str(time_execution)
		#print('tiempo: {} secs'.format(time.time()-tic))
		array_1d=results[0]

		maximo=np.argmax(array_1d)
		print(maximo)

		label = ""

		if maximo == 0:
			label = "apple"
		elif maximo == 1:
			label = "banana"
		else:
			label = "orange"
		
		detected_fruit= [(label,time_execution)]
		return detected_fruit	







