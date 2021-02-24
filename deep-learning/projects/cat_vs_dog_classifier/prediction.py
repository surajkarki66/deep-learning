from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.preprocessing.image import img_to_array


def load_image(filename):
    img = load_img(filename, target_size=(224, 224, 3))
    img = img_to_array(img)
    img = img.reshape(1, 224, 224, 3)
    img = img.astype('float32')
    img = img - [123.68, 116.779, 103.939]
    
    return img

def run(filepath, model_path=None):
    img = load_image(filepath)
    model = load_model(model_path)
    result = model.predict(img)

    if result[0] == 1:
        print("Dog")
    else:
        print("Cat")


if __name__ == "__main__":
    filename = '234.jpg'
    run(filename)
