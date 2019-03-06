from flask import Flask, request, render_template
from wtforms import Form, StringField, BooleanField, validators, FloatField
import pandas as pd
from scipy.stats import percentileofscore
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score

app = Flask(__name__)

class ModelForm(Form):
	height = FloatField('Height', [validators.DataRequired(),
									validators.NumberRange(min=0, max=1e2, message=None)])
	weight = FloatField('Weight', [validators.DataRequired(),
									validators.NumberRange(min=0, max=1e3, message=None)])
	metric = BooleanField('Use Metric')

def meters_to_inches(m):
	return float(m) * 39.37

def inches_to_meters(inch):
	return float(inch) / 39.37

def kgs_to_lbs(kg):
	return float(kg) * 2.205

def lbs_to_kgs(lbs):
	return float(lbs) / 2.205

def bmi(height, weight, metric=False):
	if metric:
		return weight / (height**2)
	return lbs_to_kgs(weight) / (inches_to_meters(height)**2)

def train(seed=None):
	df = pd.read_csv("./data/weight-height.csv",
					 header=0,
					 doublequote=True)
	df["Gender"] = df["Gender"].apply(lambda x: 1. if x == "Male" else 0.)

	x_train, x_test, y_train, y_test = train_test_split(df[["Height", "Weight"]], df["Gender"],
														test_size=0.33,
														random_state=seed)
	lr = LogisticRegression(solver="lbfgs")
	lr.fit(x_train, y_train)

	preds = lr.predict(x_test)
	acc, prec, rec = accuracy_score(y_test, preds), \
					 precision_score(y_test, preds), \
					 recall_score(y_test, preds)

	model_perf = {
		"Accuracy": "{:.02f}".format(acc),
		"Precision": "{:.02f}".format(prec),
		"Recall": "{:.02f}".format(rec)
	}

	return {"model": lr, "model_perf": model_perf}

def score(height, weight):
	trained_model = train()
	model = trained_model["model"]
	prob_male = model.predict_proba([[height, weight]])[0][1]
	is_male = True if model.predict([[height, weight]])[0] == 1 else False
	return is_male, prob_male, trained_model["model_perf"]

def descriptive_stats(height, weight):
	df = pd.read_csv("./data/weight-height.csv",
					 header=0,
					 doublequote=True)

	bmi_value = bmi(height=height, weight=weight, metric=False)

	if bmi_value >= 30:
		bmi_cat = "Obese",
	elif bmi_value >= 25:
		bmi_cat = "Overweight"
	elif bmi_value >= 18.5:
		bmi_cat = "Normal weight"
	elif bmi_value < 18.5:
		bmi_cat = "Underweight"
	else:
		bmi_cat = None

	return {
		"Percentile Male Height": percentileofscore(df[df["Gender"] == "Male"]["Height"], float(height)),
		"Percentile Male Weight": percentileofscore(df[df["Gender"] == "Male"]["Weight"], float(weight)),
		"Percentile Female Height": percentileofscore(df[df["Gender"] == "Female"]["Height"], float(height)),
		"Percentile Female Weight": percentileofscore(df[df["Gender"] == "Female"]["Weight"], float(weight)),
		"Body Mass Index (BMI)": round(bmi_value, 2),
		"BMI Category": bmi_cat
	}


@app.route("/", methods=['GET', 'POST'])
def main():
	form = ModelForm(request.form)
	if request.method == 'POST' and form.validate():
		height = float(form.height.data)
		weight = float(form.weight.data)
		if form.metric.data:
			height = meters_to_inches(height)
			weight = kgs_to_lbs(weight)
		scores = score(height=height, weight=weight)
		guess_male = True if round(scores[1], 4) * 100 >= 50. else False
		prob_male = "{:0.02f}".format(round(scores[1], 4) * 100)
		prob_female = "{:0.02f}".format(round(1 - scores[1], 4) * 100)
		model_perf = scores[2]
		stats = descriptive_stats(height=height, weight=weight)
		return render_template("model.html",
							   guess_male=guess_male,
							   prob_male=prob_male,
							   prob_female=prob_female,
							   form=form, 
							   stats=stats, 
							   model_perf=model_perf)
	return render_template("model.html", form=form)


if __name__ == "__main__":
	app.run()
