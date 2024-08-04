from flask import Flask, render_template, request
from keras.models import load_model
from keras.preprocessing import image
import numpy as np
from PIL import Image
import base64
from io import BytesIO

app = Flask(__name__)


models = {
    'apple': './Models/apple.h5',
    'cherry': './Models/cherry.h5',
    'corn': './Models/corn.h5',
    'grape': './Models/grape.h5',
    'peach': './Models/peach.h5' ,
    'pepper': './Models/pepper.h5' ,
    'potato': './Models/potato.h5' ,
    'rasoberry': './Models/rasoberry.h5' ,
    'tomato': './Models/tomato.h5' 
}

dictionaries = {
    'apple': {0: 'apple_sab', 1: 'Black_rot' , 2 : 'Cedar_apple_rust' , 3 : "Healthy" },
    'cherry': {0: 'healthy', 1: 'healthy'},
    'corn': {0: 'not healthy', 1: 'healthy'},
    'grape': {0: 'not healthy', 1: 'healthy'},
    'peach': {0: 'not healthy', 1: 'healthy'},
    'pepper': {0: 'not healthy', 1: 'healthy'},
    'potato': {0: 'not healthy', 1: 'healthy'},
    'raspberry': {0: 'not healthy', 1: 'healthy'},
    'tomato': {0: 'not healthy', 1: 'healthy'}
}



def load_selected_model(selected_option):
    model_path = models.get(selected_option)
    if model_path:
        return load_model(model_path)
    else:
        raise ValueError("Invalid model selection")

def predict_label(model, img , selected_option):
    img = img.resize((100, 100))  # Resize image to match model input size
    img = np.array(img) / 255.0  # Normalize pixel values
    img = img.reshape(1, 100, 100, 3)  # Reshape for model input
    predict = model.predict(img)
    predicted_label_index = np.argmax(predict, axis=1)[0]
    predicted_label = dictionaries[selected_option][predicted_label_index]
    accuracy = predict[0][predicted_label_index] * 100  # Confidence level as a percentage

    return [predicted_label, accuracy, selected_option]


    

# routes
@app.route("/", methods=['GET', 'POST'])
def main():
    return render_template("index.html")

@app.route("/about")
def about_page():
    return "Please subscribe  Ankit Raj WEbsite Veg Care..!!!"

@app.route("/submit", methods=['POST'])
def get_output():
    if request.method == 'POST':
        selected_option = request.form['model_option']
        selected_model = load_selected_model(selected_option)

        file = request.files['my_image']
        img = Image.open(BytesIO(file.read()))  # Open image from bytes
        prediction_label , accuracy , selected_option = predict_label(selected_model, img, selected_option)

        img_io = BytesIO()
        img.save(img_io, 'PNG')
        img_io.seek(0)
        img_base64 = base64.b64encode(img_io.getvalue()).decode('utf-8')

        print("Base64 Image:", img_base64)  # Debugging: print base64 string

    return render_template("index.html", prediction=prediction_label,accuracy = accuracy , selected_option = selected_option ,  image=img_base64)

if __name__ == '__main__':
    app.run(debug=True)
