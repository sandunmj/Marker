from flask import Flask, request
import base64
from mcq import correcting
import json

app = Flask(__name__)

@app.route('/mark', methods=['GET', 'POST'])
def mark():
    strRec = request.json['image']
    img = base64.b64decode(strRec)
    markedImage = json.dumps(correcting(img))
    print(markedImage)
    return markedImage


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8000)
