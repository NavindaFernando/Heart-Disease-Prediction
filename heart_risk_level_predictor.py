from flask import Flask,render_template,request
import joblib
import numpy as np

# load saved model
model=joblib.load('heart_risk_prediction_regression_model.sav')

# app created
app=Flask(__name__)

# end point creation
# home page
@app.route('/')
def index():

	return render_template('patient_details.html')

# route http://127.0.0.1:5000/getresults
@app.route('/getresults',methods=['POST'])
def getresults():

	# get form details
	result=request.form 

	# get dictionary keys in the form & its type is text (string) so it is converted to float
	name=result['name']
	gender=float(result['gender'])
	age=float(result['age'])
	tc=float(result['tc'])
	hdl=float(result['hdl'])
	sbp=float(result['sbp'])
	smoke=float(result['smoke'])
	bpm=float(result['bpm'])
	diab=float(result['diab'])

	# the features to be added to the model are created as an array & convert to 2d array
	test_data=np.array([gender,age,tc,hdl,smoke,bpm,diab]).reshape(1,-1)

	# the test_data is put into the loaded model
	prediction=model.predict(test_data)

	# the received prediction is put into a dictionary
	# we get the prediction as a 2d array. Therefore, the value is rounded to two decimal places
	resultDict={"name":name,"risk":round(prediction[0],2)}
	
	# the prediction is sent to the patient_results.html file
	# return resultDict
	return render_template('patient_results.html',results=resultDict)

app.run(debug=True)