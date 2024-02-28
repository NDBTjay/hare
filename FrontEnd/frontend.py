from unittest import result
from flask import Flask, redirect, render_template, request, json
import time
import subprocess
import shlex
import sys, os
import re
from ResNet50_img_preprocess import resnet50_img_preprocess
from sqnet_img_preprocess import sqnet_img_preprocess

k = " k=12" 
ell = " ell=41" 
nt = " nt=4" 
ip = " ip=127.0.0.1" 
p = " p=12345"


app = Flask(__name__)

@app.route("/")
def deal():
    return redirect("/index")

@app.route("/index")
def index():
    return render_template("index.html", src="/static/default.webp", return_name="未选择图片", datalist_control="test")

@app.route("/deal", methods=["POST"])
def get_info():
    setlist = "0"
    print(request.form.get("model"))
    img = request.files["image"]
    if request.form.get("model") != "ResNet50":
        setlist = "1"
    
    if img.filename=="":
        return redirect("/index")
    if os.path.exists("input") == False:
        os.makedirs('input')
    saveFilePath = os.path.join("input/", img.filename)
    img.save(saveFilePath)
    if setlist == "1":
        sqnet_img_preprocess(saveFilePath, 12)
        run_command = "cat sqnet_img_output/" + img.filename + ".inp | ../build/bin/sqnet-hare r=2" + k + ell + nt + ip + p
        # run_command = shlex.split(run_command)
        # print(run_command)
    else:
        resnet50_img_preprocess(saveFilePath, 12)
        run_command = "cat ResNet50_img_output/" + img.filename + ".inp | ../build/bin/sqnet-hare r=2" + k + ell + nt + ip + p
        # run_command = shlex.split(run_command)
        # print(run_command)

    output = subprocess.Popen(run_command, shell=True, stdout=subprocess.PIPE)
    testreturn = output.stdout.read()
    testreturn = testreturn.decode("UTF-8")
    # print(testreturn)
    result = re.search("Total time taken = .*", testreturn, flags=re.DOTALL).group(0)
    return render_template("index.html", result_data=result, src="static/img.jpeg", return_name="本次结果对应图片"+img.filename, datalist_control=setlist)

if __name__=="__main__":
    app.run()