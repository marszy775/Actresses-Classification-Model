from flask import Flask, jsonify, request
from flask_cors import CORS
import util

app = Flask(__name__)
CORS(app)


@app.route('/classify_image', methods=['POST'])
def classify_image():
    image_data = request.form['image_data']

    #for base64 input in postman app
    # file = request.files['img']
    # img = file.read().decode('utf-8')

    response = jsonify(util.classify_image(image_data))
    response.headers.add('Access-Control-Allow-Origin', '*')
    return response


if __name__ == "__main__":
    print("Starting Python Flask Server For Sports Celebrity Image Classification")
    util.load_saved_artifacts()
    app.run()
