from flask import Flask,request,jsonify
import text_extraction_model
from PIL import Image

app =Flask(__name__)

@app.route('/extract_text',methods=['POST'])

def extract_text():
    image = request.files['image']
    # img = Image.open(image)
    # max_size = (800, 800)  # Adjust this size as needed
    # img.thumbnail(max_size, Image.AFFINE)
    extracted_text = text_extraction_model.extract_text(image)
    return jsonify({'result':extracted_text})

if __name__ == '__main__':
    port = 8000
    app.run(host='0.0.0.0', port=port)