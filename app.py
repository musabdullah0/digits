from flask import (Flask,
                   render_template,
                   url_for, jsonify,
                   make_response,
                   request)
import urllib.request
from number_reader import Reader


app = Flask(__name__)


@app.route('/')
def home():
    return render_template('draw.html')


@app.route('/guess', methods=['POST'])
def guess():
    req = request.get_json()
    img = req['img']
    response = urllib.request.urlopen(img)
    with open('number.png', 'wb') as f:
        f.write(response.file.read())

    reader = Reader('model.pt', 'number.png')
    digit_guess = reader.read()
    print(digit_guess)

    res = make_response(jsonify({"guess": digit_guess}), 200)

    return res


# if __name__ == '__main__':
#     app.run(debug=True)
