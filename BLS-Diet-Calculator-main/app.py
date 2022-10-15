from flask import Flask, render_template,request

app = Flask(__name__)


@app.route('/')
def home():
    return render_template('home.html')


@app.route('/index')
def index():
    return render_template('index.html')




@app.route('/base',methods=['GET', 'POST'])
def calculate():
    bmi=''
    if request.method == 'POST' and 'weight' in request.form and 'weight' in request.form:
      Weight=float(request.form['weight'])
      Height=float(request.form['height'])
      bmi=round(Weight/((Height/100)**2),2)
    return render_template('base.html',bmi=bmi)


if __name__ == '__main__':
    app.run(debug=True)
