# Author: Yonghyeon Cho
# Team: team26 - Far From Home
# Creation Date: Feb/04/2022

from flask import Flask		# import Flask for the server implementation.
import recommend as rc

app = Flask(__name__)		# declare a Flask object (parameter: application package name)

###	Call the recommend.py function ###
model = rc.SVDpp_loader("model_path")		# load the model!
### ============================== ###

@app.route('/recommend/<user_id>')
def Get(user_id):		# return value: a comma separated string.
	"""
	Handle the GET request.
	"""
	result = RunModel(user_id)
    
	return result

def RunModel(user_id):
	rec_list = rc.recommend(model, user_id)
	rec_list = ','.join([str(score) for score, _ in rec_list])
	return rec_list


if __name__ == "__main__":
	app.run(debug=True, host='128.2.205.127', port=8082)