from flask import Flask
from flask import request, jsonify
import pickle
import pandas as pd
from flask_cors import CORS

app = Flask(__name__)
CORS(app)
 
@app.route("/", methods=['POST'])
def hello():
	#[1,85,66,29,0,26.6,0.351,31]					
	NumTimesPrg = request.form.get('NumTimesPrg')
	PlGlcConc = request.form.get('PlGlcConc')
	BloodP = request.form.get('BloodP')
	SkinThick = request.form.get('SkinThick')
	TwoHourSerIns = request.form.get('TwoHourSerIns')
	BMI = request.form.get('BMI')
	DiPedFunc = request.form.get('DiPedFunc')
	Age = request.form.get('Age')

	data = [NumTimesPrg,PlGlcConc,BloodP,SkinThick,TwoHourSerIns,BMI,DiPedFunc,Age]
	
	# load the model from disk
	filename = 'models/diabetes1.model'
	model = pickle.load(open(filename, 'rb'))

	# load the scaler from disk
	filename = 'models/scaler'
	scaler = pickle.load(open(filename, 'rb'))

	# We create a new (fake) person having the three most correated values high
	####new_df = pd.DataFrame([[6, 168, 72, 35, 0, 43.6, 0.627, 65]])
	new_df = pd.DataFrame([data])
	# We scale those values like the others
	new_df_scaled = scaler.transform(new_df)

	# We predict the outcome
	prediction = model.predict(new_df_scaled)

	return str(prediction[0])
 
if __name__ == "__main__":
    app.run()