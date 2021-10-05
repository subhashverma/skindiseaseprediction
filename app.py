from flask import Flask, request, render_template
from werkzeug.utils import secure_filename
from tensorflow import keras
from PIL import Image
import numpy as np
import pandas as pd
import os

app = Flask(__name__)

@app.route('/')
def hello_world():
    return render_template("image_detector.html")
	
classes =  {4: ('nv', ' melanocytic nevi'),
			6: ('mel', 'melanoma'),
			2: ('bkl', 'benign keratosis-like lesions'), 
			1: ('bcc' , ' basal cell carcinoma'),
			5: ('vasc', ' pyogenic granulomas and hemorrhage'),
			0: ('akiec', 'Actinic keratoses and intraepithelial carcinomae'),
			3: ('df', 'dermatofibroma')}
metadata = pd.read_csv("HAM10000_metadata - Copy.csv")

@app.route('/predict',methods=['POST','GET'])
def predict():
    
	fname=''	
	predop=''
	classs = classes[0]
	actual = ''
	accu = ''
	if request.method == 'POST':
		# Get the file from post request
		f = request.files['file']
		fname = f.filename
		print("->file :",fname)
		basepath = os.path.dirname(__file__)+'\static'
		print("basepath",basepath)
		file_path = os.path.join(
			basepath,secure_filename(fname))
		f.save(file_path)
		
		loaded_model = keras.models.load_model("model.keras")
		
		im1 = Image.open(basepath+"/"+fname,'r')
		im = im1.resize((28, 28))
		pix_val = list(im.getdata())
		pix_val_flat = [x for sets in pix_val for x in sets]
		Data1 = np.array(pix_val_flat).reshape(28,28,3)
		y_pred1 = loaded_model.predict([[Data1]])
		accu = list(y_pred1[0])[0]
		
		flag = 0
		for i in range(1, len(list(y_pred1[0]))):
			if accu < list(y_pred1[0])[i]:
				accu = list(y_pred1[0])[i]
				flag = 1
				classs = classes[i]
		
		#predop = "pred: " ,classs, "Actual: " ,list(metadata[metadata['image_id'] == fname.split('.')[0]]['dx'])[0], "Accuracy: " ,accu
		#print(predop)
		actual = list(metadata[metadata['image_id'] == fname.split('.')[0]]['dx'])[0]
		print("Predicted class =", classs)
		print("Actual class =", actual)
		print("Accuracy =", accu)

	app.config["CACHE_TYPE"] = "null"
	
	return render_template("image_detector.html", filename = fname, pred = classs, pred1 = actual, pred2 = accu)


if __name__ == '__main__':
    app.run(debug=True)
