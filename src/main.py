from flask import Flask, request, redirect, render_template, send_from_directory
from werkzeug.utils import secure_filename
from io import BytesIO
import numpy as np
import os
app = Flask(__name__)
app.config['API_VERSION'] = '1.0.0'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024
app.config['ALLOWED_EXTENSIONS'] = {'dat'}

np.set_printoptions(threshold=np.nan)

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1] in app.config['ALLOWED_EXTENSIONS']


@app.route('/', methods=['GET'])
def default_documentation():
    return render_template('html/docs.html',
                           version=app.config['API_VERSION']
                           )


@app.route('/config', methods=['GET'])
def config():
    return render_template('html/config.html',
                           version=app.config['API_VERSION'],
                           maxContentLength=str(app.config['MAX_CONTENT_LENGTH']),
                           allowedExtensions=str(app.config['ALLOWED_EXTENSIONS'])
                           )


@app.route('/ping', methods=['GET'])
def ping():
    return 'Application up and running...'


@app.route('/nrmeq', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        if ('X' or 'y') not in request.files:
            return redirect(request.url)
        Xf = request.files['X']
        yf = request.files['y']
        if Xf and allowed_file(Xf.filename):
            Xc = BytesIO(Xf.read())
            yc = BytesIO(yf.read())
            Xp = np.loadtxt(Xc,dtype=float,ndmin=2)
            Xpr, Xpc = Xp.shape
            X = np.c_[np.ones(Xpr), Xp]
            y = np.loadtxt(yc,dtype=float,ndmin=2)
            Xn = nrm(X)
            return pretty_pad(str(nrmeq(Xn,y)))
        return "An error has occurred while uploading the file"
      
      
def nrmeq(X,y):
    Xam = np.asmatrix(X)
    Xt = Xam.transpose()
    yam = np.asmatrix(y)
    return np.linalg.pinv(Xt.dot(X)).dot(Xt).dot(yam)
      

def nrm(X):
    tr,tc = X.shape
    nX = np.ones([tr,tc])
    for c in range(1,tc-1):
        std = np.std(X[:,c])
        min = np.min(X[:,c])
        for r in range(0,tr-1):
            nX[r-1,c-1] = (X[r-1,c-1] - min) / std 
    return nX
        
        
def pretty_pad(string):
    return "\n\n" + string + "\n\n"
