
#import the neccessary Header files
import tensorflow as tf
from tensorflow.keras import models, layers
import matplotlib.pyplot as plt
import os

BATCH_SIZE = 32
IMAGE_SIZE = 256
CHANNELS=3
EPOCHS=50

class CreateModel:
   
    # Sample Method 
    def preprocessing(self):
        
        dataset = tf.keras.preprocessing.image_dataset_from_directory(
    	"dataset",
    	seed=123,
    	shuffle=True,
    	image_size=(IMAGE_SIZE,IMAGE_SIZE),
    	batch_size=BATCH_SIZE
	)
    def SplitDataset(self):
        
        train_ds = dataset.take(int(len(dataset)*0.8))
        test_ds = dataset.skip(int(len(dataset)*0.8))
        val_ds = test_ds.take(int(len(test_ds) *0.6))
        test_ds = test_ds.skip(int(len(test_ds) *0.6))  
        train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=tf.data.AUTOTUNE)
        val_ds = val_ds.cache().shuffle(1000).prefetch(buffer_size=tf.data.AUTOTUNE)
        test_ds = test_ds.cache().shuffle(1000).prefetch(buffer_size=tf.data.AUTOTUNE)
  
    def DataAugmentation(self):
        
        resize_and_rescale = tf.keras.Sequential([
  	 layers.experimental.preprocessing.Resizing(IMAGE_SIZE, IMAGE_SIZE),
      	 layers.experimental.preprocessing.Rescaling(1./255),
	 ])
        data_augmentation = tf.keras.Sequential([
  	 layers.experimental.preprocessing.RandomFlip("horizontal_and_vertical"),
  	 layers.experimental.preprocessing.RandomRotation(0.2),
	 ])
        train_ds = train_ds.map(
    		  lambda x, y: (data_augmentation(x, training=True), y)
		   ).prefetch(buffer_size=tf.data.AUTOTUNE)

    def Model(self):
        
        input_shape = (BATCH_SIZE, IMAGE_SIZE, IMAGE_SIZE, CHANNELS)
        n_classes = 3

        model = models.Sequential([
        resize_and_rescale,
        layers.Conv2D(32, kernel_size = (3,3), activation='relu', input_shape=input_shape),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64,  kernel_size = (3,3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64,  kernel_size = (3,3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Flatten(),
        layers.Dense(64, activation='relu'),
        layers.Dense(n_classes, activation='softmax'),
        ])

        model.build(input_shape=input_shape)
        model.compile(
        optimizer='adam',
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
        metrics=['accuracy']
        )
        history = model.fit(
        train_ds,
        batch_size=BATCH_SIZE,
        validation_data=val_ds,
        verbose=1,
        epochs=50,
        )
    def SaveModel(self):
        directory = os.getcwd()
        model.save(directory)            
        
    	 
if __name__ == "__main__":
    
    create =  CreateModel()
    create.preprocessing ()
    create.SplitDataset () 	 
    create.DataAugmentation ()	 
    create.Model () 
    create.SaveModel () 




   	
