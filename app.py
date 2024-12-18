from flask import Flask, request, jsonify
from flask_restful import Resource, Api
import tensorflow as tf
from property_valuation import load_and_preprocess_image

app = Flask(__name__)
api = Api(app)

# Load pretrained model
model = tf.keras.models.load_model('property_valuation_model.h5')

class PropertyValuation(Resource):
    def post(self):
        file = request.files['image']
        image = load_and_preprocess_image(file)
        image = np.expand_dims(image, axis=0)  # Add batch dimension
        
        predicted_price = model.predict(image)
        return jsonify({'predicted_price': str(predicted_price[0][0])})

api.add_resource(PropertyValuation, '/predict_value')

if __name__ == '__main__':
    app.run(debug=True)
