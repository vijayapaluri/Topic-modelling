

from flask import Flask,render_template,request,url_for
import json
from werkzeug.utils import secure_filename
import os
import data_preprocess as preprocess
import pandas as pd

UPLOAD_FOLDER = 'C:/Users/Vijaya/Topic_modelling_project/uploads'



app=Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
@app.route('/',methods=["GET","POST"])
def home():
    if request.method=="POST":
        #convert=request.form['convert_dict']

        if "convert_vtt" in request.form:
            file=request.files['vtt_file']
            filename = secure_filename(file.filename)
            file_path=os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            context="File uploaded successfully"
            file_name=file.filename.split('.')[0]
            vtt_file=preprocess.convert_to_vtt(file_path,UPLOAD_FOLDER)
            return render_template("index.html", ctx=context)
        elif "convert_dict" in request.form:
            data,df=preprocess.preprocess_data('C:/Users/Vijaya/Topic_modelling_project/uploads/vtt_df.csv', UPLOAD_FOLDER)
            
            #json_data=json.loads(data)
            with open("C:/Users/Vijaya/Topic_modelling_project/uploads/topic.json","r") as data:
                json_data=json.load(data)
            return render_template("index.html", tables=[df.to_html(classes='data',header="true")],js_data=json_data)
    return render_template("index.html")



        

if __name__=='__main__':
    app.run(debug=True)

