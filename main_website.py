from flask import Flask, redirect, render_template, url_for
from second_notebook import *





app = Flask(__name__)





@app.route("/")
def home():
	return render_template("politics.html", content = political_list)




@app.route("/politics.html")
def politics():
	return render_template("politics.html", content = political_list)




@app.route("/business.html")
def business():
	return render_template("business.html", content = business_list)




@app.route("/sports.html")
def sports():
	return render_template("sports.html", content = sports_list)





@app.route("/entertainment.html")
def entert():
	return render_template("entertainment.html", content = enter_list)





@app.route("/tech.html")
def tech():
	return render_template("tech.html", content = tech_list)




@app.route("/other.html")
def other():
	return render_template("other.html", content = other_list)





if __name__ == "__main__":
	app.debug = True
	app.run()