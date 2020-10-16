from flask import Flask, render_template, request, url_for
import pickle
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
import joblib


#cv = pickle.load(open('cv.pkl','rb')) #loading the cv picke file 
app = Flask(__name__) 


@app.route('/')
def home():
	return render_template('home.html')


@app.route('/predict', methods=['POST'])
def predicting():
	NBmodel = open("SpamHam_model.pkl", "rb")
	model=joblib.load(NBmodel)

	cvmodel=open("cv.pkl", "rb")
	cv=joblib.load(cvmodel)

	if request.method == 'POST':
		message = request.form['message']
		data = [message]
		vect = cv.transform(data).toarray()
		to_predict = model.predict(vect)


	return render_template('result.html', prediction = to_predict)

if __name__ == '__main__':
	app.run(debug=True)




	#message = request.form['message']
	#data = [message]
	#vect = np.transform(data).toarray()
	#my_prediction = model.predict(vect)
	#return render_template('result.html', prediction=my_prediction)





