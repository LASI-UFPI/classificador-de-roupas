from PIL import Image, ImageOps
import numpy as np
from tensorflow.keras import models

def predictClothes(filenameImage, filenameModel):
  image = Image.open(filenameImage)
  image = image.resize((28,28))
  image = ImageOps.grayscale(image)
  image = np.array(image, ndmin=3)/255

  model = models.load_model(filenameModel)

  result_predict = int(np.argmax(model.predict(image), axis=-1))
  
  name_classes = ["Camiseta", "calça", "Suéter", "Vestido", "Casaco","Sandália","Camisa","Tênis", "Bolsa","Botas"]
  result = str(name_classes[result_predict])
  return result

