from flask import Flask, request, render_template
from wtforms import Form, BooleanField, validators, FloatField
import numpy as np
from scipy.stats import percentileofscore
from pypmml import Model

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


def score(height, weight, model_path="height_weight_lr_model.pmml", threshold=0.5):
	fitted_model = Model.load(model_path)
	pred = fitted_model.predict({"Height": height, "Weight": weight})
	prob_male = float(pred.get("probability(1.0)"))
	prob_female = float(pred.get("probability(0.0)"))
	guess_male = True if prob_male >= threshold else False
	return guess_male, prob_male, prob_female


def descriptive_stats(height, weight):
	df = np.loadtxt(
		fname="./data/weight-height.csv",
		dtype=np.str,
		skiprows=1,
		delimiter=","
	)

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
		"Percentile Male Height": percentileofscore(df[df[:, 0] == "\"Male\""][:, 1].astype(np.float32), float(height)),
		"Percentile Male Weight": percentileofscore(df[df[:, 0] == "\"Male\""][:, 2].astype(np.float32), float(weight)),
		"Percentile Female Height": percentileofscore(df[df[:, 0] == "\"Female\""][:, 1].astype(np.float32), float(height)),
		"Percentile Female Weight": percentileofscore(df[df[:, 0] == "\"Female\""][:, 2].astype(np.float32), float(weight)),
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
		guess_male, prob_male, prob_female = score(height=height, weight=weight)
		prob_male_str = f"{prob_male*100:0.02f}"
		prob_female_str = f"{prob_female*100:0.02f}"
		model_perf = {'Accuracy': '0.91', 'Precision': '0.91', 'Recall': '0.91'}
		stats = descriptive_stats(height=height, weight=weight)
		return render_template(
			"model.html",
			guess_male=guess_male,
			prob_male=prob_male_str,
			prob_female=prob_female_str,
			form=form,
			stats=stats,
			model_perf=model_perf
		)
	return render_template("model.html", form=form)


if __name__ == "__main__":
	app.run()
