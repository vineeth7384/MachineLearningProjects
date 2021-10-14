# test.py ---- To Test the saved model
import cv2

class TestModel:
    
    def ValidateImage(self):
        # Replace test image path
        img = cv2.imread("/content/PlantVillage/Potato___Late_blight/01ad74ce-eb28-42c7-9204-778d17cfd45c___RS_LB 2669.JPG")
        directory = os.getcwd()
        new_model = tf.keras.models.load_model(directory)
        image_array = tf.keras.utils.img_to_array(
        img, data_format=None, dtype=None
        )
        image_array  = tf.expand_dims(image_array , 0)

        batch_prediction = new_model.predict(image_array)
        print("predicted label:",np.argmax(batch_prediction[0]))            	

if __name__ == "__main__":
    
    Test =  TestModel()
    Test.ValidateImage()
    
