from flask import Flask,request,render_template
import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler
from src.pipeline.prediction_pipeline import CustomData,PredictPipeline

application=Flask(__name__)

app=application

## Route for a home page

@app.route('/')
def index():
    return render_template('index.html') 


@app.route('/predictdata',methods=['GET','POST'])
def predict_datapoint():

    if request.method=='GET':
        return render_template('home.html')
    else:

        bt=request.form.get('batting_team'),
        batting_team= bt[0]
        # print(batting_team, type(batting_team))
        bowling_team=request.form.get('bowling_team'),
        bowling_team=bowling_team[0]
        ct=request.form.get('city'),
        city=ct[0]

        current_score=request.form.get('current_score'),
        over=request.form.get('over'),
        wl=request.form.get('wickets_left'),
        # print('this is wl ',wl)
        wickets_left= (int(wl[0])),
        # print('this is wicket_left ',wickets_left, "type is ", type(wickets_left))
        lf=request.form.get('last_five'),
        last_five=int(lf[0]),
        balls_left = 120 - (int(over[0])*6)
        crr = int(current_score[0])/int(over[0])

        current_score=int(current_score[0]),

        # print(batting_team,bowling_team,city,current_score,over,wickets_left,last_five,balls_left,crr)
        # cd = batting_team[0],bowling_team[0],city[0],int(current_score[0]),int(over[0]),int(wicket[0]),int(last_five[0]),balls_left,crr

        data = CustomData(batting_team,bowling_team,city,current_score,wickets_left,last_five,balls_left,crr)

        pred_df = data.get_data_as_data_frame()
        print(pred_df)
        predict_pipeline = PredictPipeline()
        result = predict_pipeline.predict(pred_df)
        
        return render_template('home.html',result=result[0])
    

if __name__=="__main__":
    app.run(host="0.0.0.0")    




# batting_team=request.form.get('batting_team'),
# bowling_team=request.form.get('bowling_team'),
# city=request.form.get('city'),
# current_score=request.form.get('current_score'),
# over=request.form.get('over'),
# wicket=request.form.get('wicket'),
# last_five=request.form.get('last_five'),
# # wicket=float(request.form.get('wicket')),
# # last_five=float(request.form.get('last_five'))
# balls_left = 120 - (over*6)
# crr = current_score/over