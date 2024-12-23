from flask import Flask, render_template, request, Response
from flask_wtf import FlaskForm
from wtforms import StringField, SubmitField
import pickle
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem, Descriptors
from rdkit.ML.Descriptors import MoleculeDescriptors
from rdkit.Chem.Fingerprints import FingerprintMols
#import chembl_structure_pipeline
import numpy as np
import pandas as pd
from rdkit.Chem import PandasTools
import joblib
import pickle
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from molvs import standardize_smiles
from rdkit.Chem import Draw
import io
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
import base64
from rdkit.Chem.Draw import SimilarityMaps
from matplotlib import cm
from PIL import Image 
from ochem import mycalc




import warnings
warnings.filterwarnings('ignore')

best_clf_GBM=pickle.load(open('fxr_svc_fcfp4.pkl','rb'))

app = Flask(__name__)
app.config["SECRET_KEY"] = "mysecret"

class inputform(FlaskForm):
      #c1n=StringField("Compound 1 Nature")
      c1a=StringField("Put your input SMILES for prediction")
      submit=SubmitField("Submit")

def rdkit_numpy_convert(fp_vs):
    output = []
    for f in fp_vs:
        arr = np.zeros((1,))
        DataStructs.ConvertToNumpyArray(f, arr)
        output.append(arr)
    return np.asarray(output)


def convert(df):
    ls=[]
    for smile in df['Canonical SMILES']:
        smiles=standardize_smiles(smile)
        ls.append(Chem.MolFromSmiles(smile))
    fp = [AllChem.GetMorganFingerprintAsBitVect(m, radius=radius,nBits=nBits,useFeatures=useFeatures,useChirality = True) for m in ls]
    x = rdkit_numpy_convert(fp)
    return x

def process_fingerprint(smi,best_clf_GBM):
    mol=Chem.MolFromSmiles(smi)
    f_vs = [AllChem.GetMorganFingerprintAsBitVect(mol, radius=2,nBits=1024,useFeatures=True,useChirality = True)]
    X = rdkit_numpy_convert(f_vs)
    prediction_GBM = best_clf_GBM.predict(X)
    prediction_GBM = np.array(prediction_GBM)
    prediction_GBM = np.where(prediction_GBM == 1, "Active", "Inactive")
    print(prediction_GBM)
    return mol,prediction_GBM[0]

def make_image(smi):
    mol=Chem.MolFromSmiles(smi)
    img=Draw.MolToImage(mol)
    return img

def fpFunction(m, atomId=-1):
    fp = SimilarityMaps.GetMorganFingerprint(m, atomId=atomId, radius=2, nBits=1024,useChirality = True)
    return fp

def getProba(fp, predictionFunction):
    return predictionFunction((fp,))[0][1]

@app.route("/", methods=["GET","POST"])
def index():
    smiles = None
    activity = None
    img_data = None
    img_data2 = None
    error_message = None
    input=inputform()
    #print("Hello")
    strn=''
    act=''
    if request.method == 'POST':
       print(str(request.form['c1a']))
       mol,pred=process_fingerprint(request.form['c1a'],best_clf_GBM)
       strn=strn+str(pred)
       global fig
       fig = plt.figure(figsize=(100, 100), dpi=100)
       #ax = fig.add_subplot(111)
       fig,_=SimilarityMaps.GetSimilarityMapForModel(mol, fpFunction, lambda x: getProba(x, best_clf_GBM.predict_proba), colorMap=cm.PiYG_r)
       #print(img_data)
       #plt.tight_layout()
       img_buf = io.BytesIO()
       fig.savefig(img_buf, format='png',bbox_inches='tight')
       im = Image.open(img_buf)
       output = io.BytesIO()
       im.save(output, format="PNG")
       img_data = f"data:image/png;base64,{base64.b64encode(output.getvalue()).decode('utf-8')}"
       act=act+str(mycalc('fatimaBest.pickle',request.form['c1a'])[0])
       fig2=mycalc('fatimaBest.pickle',request.form['c1a'])[1]
       img_data2 = fig2     
    return render_template('index.html', template_form=input, template_list=strn, img_data=img_data,act=act, img_data2=img_data2)


'''@app.route('/plot.png')
def plot_png():
    #plt.savefig('scattervv2.svg', bbox_inches='tight') 
    img_buf = io.BytesIO()
    fig.savefig(img_buf, format='png',bbox_inches='tight')
    im = Image.open(img_buf)
    #im.show(title="My Image")
    output = io.BytesIO()
    im.save(output, format="PNG")
    img_data = f"data:image/png;base64,{base64.b64encode(output.getvalue()).decode('utf-8')}"
    FigureCanvas(im).print_png(output)
    return Response(output.getvalue(), mimetype='image/png')'''

if __name__=="__main__":
   app.run(debug=True)
