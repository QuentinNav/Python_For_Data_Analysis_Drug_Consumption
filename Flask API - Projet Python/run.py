# -*- coding: utf-8 -*-
"""
Created on Wed Dec 22 14:57:24 2021

@author: bruno
"""
from flask import Flask, render_template, request

app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def form():
    if request.method == 'POST': 
        print(request.form.getlist('mycheckbox'))
        print(request.form['nscore'])
        print(request.form['escore'])
        print(request.form['oscore'])
        print(request.form['ascore'])
        print(request.form['cscore'])
        return 'Done'
    return render_template('form.html')


if __name__ == '__main__':
    app.run(host='localhost', port=5000)