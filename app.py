from flask import Flask,request,render_template
from src.pipeline.predict_pipeline import CustomData,PredictPipeline

app=Flask(__name__)

@app.route('/')
def index():
    return render_template('home.html',results=None)

@app.route('/predictdata',methods=['GET',"POST"])
def predict_datapoint():
    
    if request.method=="GET":
        return render_template('home.html',results=None)
    else:
        data=CustomData(
            Pregnancies=request.form.get('Pregnancies'),
            Glucose=request.form.get('Glucose'),
            BloodPressure=request.form.get('BloodPressure'),
            SkinThickness=request.form.get('SkinThickness'),
            Insulin=request.form.get('Insulin'),
            BMI=request.form.get("BMI"),
            DiabetesPedigreeFunction=request.form.get('DiabetesPedigreeFunction'),
            Age=request.form.get('Age')   
        )
        pred_df=data.data_frame()
        print(pred_df)
        
        predict_pipeline=PredictPipeline()
        results=predict_pipeline.predict(pred_df)
        
        result_value=float(results[0])
        print("Prediction:", result_value)
        return render_template('home.html',results=result_value)
    
    
if __name__=="__main__":
    app.run(host='0.0.0.0',debug=True,port=5001)    