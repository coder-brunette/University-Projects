#from flask import Flask, request, jsonify, render_template
import os
#from flask_cors import CORS, cross_origin
from com_in_ai_utils.utils import decodeImage
from predict import crack_detection

os.putenv('LANG', 'en_US.UTF-8')
os.putenv('LC_ALL', 'en_US.UTF-8')

#app = Flask(__name__)
#CORS(app)




# @cross_origin()
class ClientApp:
    def __init__(self):
        self.filename = "newlatest/input_shape.jpg"
        self.classifier = crack_detection(self.filename)

'''
@app.route("/", methods=['GET'])
@cross_origin()
def home():
    return render_template('index.html')


@app.route("/predict", methods=['POST'])
@cross_origin()
'''
def predictRoute():
    #image = request.json['image']
    image="newlatest/input_shape.jpg"
    #print(decodeImage(image, clApp.filename))
    #decodeImage(image, clApp.filename)
    clApp.classifier.predictiondogcat()
    result = [{"image" : "Here we go"}]
    
    return "success"


clApp = ClientApp()
# #port = int(os.getenv("PORT"))
'''
if __name__ == "__main__":
    # clApp = ClientApp()
    # app.run(host='0.0.0.0', port=port)
    app.run(debug=True)
'''