import os

from Utils.utils import decodeImage
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS, cross_origin

from predict import retinopathy

os.putenv('LANG', 'en_US.UTF-8')
os.putenv('LC_ALL', 'en_US.UTF-8')

application = Flask(__name__, template_folder='templates', static_folder='assets')
app=application
CORS(app)
app.config['DEBUG'] = True


# @cross_origin()
class ClientApp:
    def __init__(self):
        self.filename = "inputImage.png"
        self.classifier = retinopathy(self.filename)


@app.route("/", methods=['GET'])
@cross_origin()
def homePage():
    return render_template('index.html')


@app.route("/predict", methods=['POST'])
@cross_origin()
def predictRoute():
    try:
        if request.method == 'POST':
            image = request.json['image']
            decodeImage(image, clintApp.filename)
            result = clintApp.classifier.predictionretinopathy()
            print(result)
            result = jsonify(result)
            scroll = "pred"
            # pred = malaria()
            # result = pred.predictionretinopathy(image)
            '''for file in os.listdir():
                if file.endswith('.jpg'):
                    os.remove(file)'''
            return result, scroll
        else:
            result = jsonify("value error")
            return render_template('results.html', result=result)
    except Exception as e:
        print('exception is   ', e)
        # return Response(e)
        return jsonify(e)


# port = int(os.getenv("PORT"))
if __name__ == "__main__":
    clintApp = ClientApp()
    host = '0.0.0.0'
    port = 5000
    app.run(debug=True)
    # httpd = simple_server.make_server(host, port, app)
    # print("Serving on %s %d" % (host, port))
    # httpd.serve_forever()
