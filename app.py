# importing all the required python modules


from matplotlib.pyplot import title
from bs4 import BeautifulSoup
from utils.disease import disease_dic
from utils.model import ResNet9
from PIL import Image
from torchvision import transforms
import torch
import io
from datetime import datetime
import urllib.parse
from utils.fertilizer import fertilizer_dic

import plotly.express as px
import plotly
import json
from flask import Flask, escape, flash, redirect, request, render_template, session, url_for, Markup
import pickle
import pandas as pd

import numpy as np
from sklearn import datasets
from flask_mongoengine import MongoEngine
from werkzeug.security import generate_password_hash, check_password_hash, gen_salt
import requests 
app = Flask(__name__)


# Mongodb Database Connection Use localHost
name = 'deepak'
username = urllib.parse.quote_plus('name')
pass1='deepakprasad'
password= urllib.parse.quote_plus('pass1')
app.secret_key = 'aifarming'
app.config["MONGODB_SETTINGS"] = {'DB': "my_app", "host":'mongodb://127.0.0.1:27017/aiagriculture'}

db = MongoEngine()
db1 = MongoEngine()
db.init_app(app)
db1.init_app(app)


class users(db.Document):
    username = db.StringField()
    email = db.StringField()
    phone = db.StringField()
    profession = db.StringField()
    password = db.StringField()
    rpassword = db.StringField()
    registered_Date = db.DateTimeField(datetime.now)


class Market(db1.Document):
    fname = db1.StringField()
    lname = db1.StringField()
    email = db1.StringField()
    phone = db1.StringField()
    address = db1.StringField()
    croptype = db1.StringField()
    quantity = db1.StringField()
    cropname = db1.StringField()
    msp = db1.StringField()
    registered_Date = db1.DateTimeField(datetime.now)

# Login Function
@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        _username = request.form['email']
        _password = request.form['password']
        users1 = users.objects(email=_username).count()
        if users1 > 0:
            users_response = users.objects(email=_username).first()
            password = users_response['password']

            if check_password_hash(password, _password):
                session['logged_in'] = users_response['username']
                flash('You were logged In')
                # page=(username_s).capitalize()
                # username_s=True
                # return  render_template('index.html',page=(username_s).capitalize())
                return redirect(url_for('home'))

            else:
                error = "Invalid Login / Check Your Username And Password"
                return render_template('login.html', errormsg=error)
        else:
            error = "No User Exists"
            return render_template('login.html', errormsg=error)

    return render_template('login.html')


# Signup Function
@app.route('/signup', methods=['GET', 'POST'])
def register():

    today = datetime.today()

    # Output message if something goes wrong...
    # Check if "username", "password" and "email" POST requests exist (user submitted form)
    if request.method == 'POST':
        # Create variables for easy access
        _username = request.form['uname']
        _email = request.form['email']
        # password = request.form['password']
        _phone = request.form['phone']
        _profession = request.form['phone']
        _password = request.form['password']
        _rpassword = request.form['rpassword']
        if _email and _username and _password:
            hashed_password = generate_password_hash(_password)
            users1 = users.objects(email=_email)
            if not users1:
                usersave = users(username=_username, email=_email, profession=_profession, phone=_phone,
                                password=hashed_password, rpassword=hashed_password, registered_Date=today)
                usersave.save()
                msg = '{"html":"OKay you have registered"}'
                msghtml = json.loads(msg)
                return msghtml["html"] and redirect('/login')
            else:
                msg = f"It seems that {_email} You have already Registered"
            
                return render_template('signup.html',msg=msg)
        else:
            msg="Please enter email address & required details"
            render_template("signup.html", msg=msg)
    return render_template("signup.html")


# logout Function
@app.route('/logout')
# @login_required
def logout():
    session.pop('logged_in', None)
    flash('You were Logout Out Sucessfully')

    return redirect('/')


# # importing and Loading Pickle File
model = pickle.load(open('./pickle/crops.pkl', 'rb'))
fert = pickle.load(open('./pickle/fertilizer1.pkl', 'rb'))


# API Based Waether data

def weather_fetch(city_name):
    """
    Fetch and returns the temperature and humidity of a city
    :params: city_name
    :return: temperature, humidity
    """
    api_key = '3b0124dbadcdbcf295fd8c009f8efc0c'
    base_url = "http://api.openweathermap.org/data/2.5/weather?"

    complete_url = base_url + "appid=" + api_key + "&q=" + city_name
    response = requests.get(complete_url)
    x = response.json()

    if x["cod"] != "404":
        y = x["main"]

        temperature = round((y["temp"] - 273.15), 2)
        humidity = y["humidity"]
        return temperature, humidity
    else:
        return None

# importing some information of crops


basePrice = {
    "paddy": 1940,
    "arhar": 6300,
    "bajra": 1250,
    "barley": 1650,
    "copra": 10590,
    "cotton": 6025,
    "sesamum": 4200,
    "gram": 5230,
    "groundnut": 5550,
    "jowar": 2758,
    "maize": 1870,
    "masoor": 5500,
    "moong": 7275,
    "niger": 6930,
    "ragi": 3295,
    "rape": 4650,
    "jute": 4500,
    "safflower": 5441,
    "soyabean": 3950,
    "sugarcane": 3390,
    "sunflower": 5885,
    "urad": 6300,
    "wheat": 2015

}


# some static data about crops
crop_data = {
    "wheat": ["/static/images/wheat.jpg", "U.P., Punjab, Haryana, Rajasthan, M.P., bihar", "rabi", "Sri Lanka, United Arab Emirates, Taiwan"],
    "paddy": ["/static/images/paddy.jpg", "W.B., U.P., Andhra Pradesh, Punjab, T.N.", "kharif", "Bangladesh, Saudi Arabia, Iran"],
    "barley": ["/static/images/barley.jpg", "Rajasthan, Uttar Pradesh, Madhya Pradesh, Haryana, Punjab", "rabi", "Oman, UK, Qatar, USA"],
    "maize": ["/static/images/maize.jpg", "Karnataka, Andhra Pradesh, Tamil Nadu, Rajasthan, Maharashtra", "kharif", "Hong Kong, United Arab Emirates, France"],
    "bajra": ["/static/images/bajra.jpg", "Rajasthan, Maharashtra, Haryana, Uttar Pradesh and Gujarat", "kharif", "Oman, Saudi Arabia, Israel, Japan"],
    "copra": ["/static/images/copra.jpg", "Kerala, Tamil Nadu, Karnataka, Andhra Pradesh, Orissa, West Bengal", "rabi", "Veitnam, Bangladesh, Iran, Malaysia"],
    "cotton": ["/static/images/cotton.jpg", "Punjab, Haryana, Maharashtra, Tamil Nadu, Madhya Pradesh, Gujarat", " China, Bangladesh, Egypt"],
    "masoor": ["/static/images/masoor.jpg", "Uttar Pradesh, Madhya Pradesh, Bihar, West Bengal, Rajasthan", "rabi", "Pakistan, Cyprus,United Arab Emirates"],
    "gram": ["/static/images/gram.jpg", "Madhya Pradesh, Maharashtra, Rajasthan, Uttar Pradesh, Andhra Pradesh & Karnataka", "rabi", "Veitnam, Spain, Myanmar"],
    "groundnut": ["/static/images/groundnut.jpg", "Andhra Pradesh, Gujarat, Tamil Nadu, Karnataka, and Maharashtra", "kharif", "Indonesia, Jordan, Iraq"],
    "arhar": ["/static/images/arhar.jpg", "Maharashtra, Karnataka, Madhya Pradesh and Andhra Pradesh", "kharif", "United Arab Emirates, USA, Chicago"],
    "sesamum": ["/static/images/sesamum.jpg", "Maharashtra, Rajasthan, West Bengal, Andhra Pradesh, Gujarat", "rabi", "Iraq, South Africa, USA, Netherlands"],
    "jowar": ["/static/images/jowar.jpg", "Maharashtra, Karnataka, Andhra Pradesh, Madhya Pradesh, Gujarat", "kharif", "Torronto, Sydney, New York"],
    "moong": ["/static/images/moong.jpg", "Rajasthan, Maharashtra, Andhra Pradesh", "rabi", "Qatar, United States, Canada"],
    "niger": ["/static/images/niger.jpg", "Andha Pradesh, Assam, Chattisgarh, Gujarat, Jharkhand", "kharif", "United States of American,Argenyina, Belgium"],
    "rape": ["/static/images/rape.jpg", "Rajasthan, Uttar Pradesh, Haryana, Madhya Pradesh, and Gujarat", "rabi", "Veitnam, Malaysia, Taiwan"],
    "jute": ["/static/images/jute.jpg", " West Bengal , Assam , Orissa , Bihar , Uttar Pradesh", "kharif", "JOrdan, United Arab Emirates, Taiwan"],
    "safflower": ["/static/images/safflower.jpg",  "Maharashtra, Karnataka, Andhra Pradesh, Madhya Pradesh, Orissa", "kharif", " Philippines, Taiwan, Portugal"],
    "soyabean": ["/static/images/soyabean.jpg",  "Madhya Pradesh, Maharashtra, Rajasthan, Madhya Pradesh and Maharashtra", "kharif", "Spain, Thailand, Singapore"],
    "urad": ["/static/images/urad.jpg",  "Andhra Pradesh, Maharashtra, Madhya Pradesh, Tamil Nadu", "rabi", "United States, Canada, United Arab Emirates"],
    "ragi": ["/static/images/ragi.jpg",  "Maharashtra, Tamil Nadu and Uttarakhand", "kharif", "United Arab Emirates, New Zealand, Bahrain"],
    "sunflower": ["sunflower.jpg",  "Karnataka, Andhra Pradesh, Maharashtra, Bihar, Orissa", "rabi", "Phillippines, United States, Bangladesh"],
    "sugarcane": ["sugarcane.jpg", "Uttar Pradesh, Maharashtra, Tamil Nadu, Karnataka, Andhra Pradesh", "kharif", "Kenya, United Arab Emirates, United Kingdom"]
}

# rendering index page


@app.route('/')
def home():
    # checking for user login
    if not session.get('logged_in'):
        return render_template("index.html")
    return render_template("index.html")

# -----------------------------------------------------------------------------------------


# Crop prediction using RandomForest


@app.route('/crop', methods=['GET', 'POST'])
def predict():
    if request.method == "POST":
        Nitrogen = int(request.form['Nitrogen'])
        Potassium = int(request.form['Potassium'])
        Phosphorous = int(request.form['Phosphorous'])
        State = request.form['stt']
        city = request.form['city']
        Rainfall = int(request.form['Rainfall'])
        PH = int(request.form['PH'])
        temperature = ''
        humidity = ''
        if weather_fetch(city) != None:
            temperature, humidity = weather_fetch(city)

        prediction = model.predict(
            np.array([[Nitrogen, Potassium, Phosphorous, temperature, humidity, PH, Rainfall]]))

        return render_template("crop.html", prediction_text="Hey Farmer You should Grow: {}".format(prediction[0]))

    else:
        return render_template("crop.html")

# ------------------------------------------------------------------------------------------------

# Fertilizer Prediction using RandomForest


@app.route('/fertilizer', methods=['GET', 'POST'])
def predict1():
    if request.method == "POST":
        moisture = int(request.form['Nitrogen'])
        Nitrogen = int(request.form['Nitrogen'])
        Potassium = int(request.form['Potassium'])
        Phosphorous = int(request.form['Phosphorous'])
        State = request.form['stt']
        city = request.form['city']
        croptype = request.form['croptype']
        soiltype = request.form['soiltype']
        crop = 0
        soil = 0
        if(croptype == 'Barley'):
            crop = 0
        elif(croptype == "Cotton"):
            crop = 1
        elif(croptype == "Ground Nuts"):
            crop = 2
        elif(croptype == "Maize"):
            crop = 3
        elif(croptype == "Millets"):
            crop = 4
        elif(croptype == "Oil seeds"):
            crop = 5
        elif(croptype == "Paddy"):
            crop = 6
        elif(croptype == "Pulses"):
            crop = 7
        elif(croptype == "Sugarcane"):
            crop = 8
        elif(croptype == "Tobacco"):
            crop = 9
        elif(croptype == "Wheat"):
            crop = 10

        if(soiltype == "Black"):
            soil = 0
        elif(soiltype == "Clayey"):
            soil = 1
        elif(soiltype == "Loamy"):
            soil = 2
        elif(soiltype == "Red"):
            soil = 3
        elif(soiltype == "Sandy"):
            soil = 4
        else:
            soil = 0

        print(croptype)

        # calling API to get real-time data

        if weather_fetch(city) != None:
            temperature, humidity = weather_fetch(city)

        Fertilizer_prediction = fert.predict(
            np.array([[temperature, moisture, humidity, soil, crop, Nitrogen, Potassium, Phosphorous]]))

        response = Markup(str(fertilizer_dic[Fertilizer_prediction[0]]))

        return render_template("fertilizer.html", prediction_text="The Suitable Fertilizer For your crop is: {}".format(Fertilizer_prediction[0]), recommendation=response)

    else:

        return render_template("fertilizer.html")


# --------------------------------------------------------------------------------------------------------


# Crop Price Prediction & Forecasting


@app.route('/forecast', methods=['GET', 'POST'])
def predict2():

    if request.method == "POST":
        cropsgraph = request.form["croptype"]

        # get required crop data csv files
        commodity_dict = {
            "arhar": "static/data/Arhar.csv",
            "bajra": "static/data/Bajra.csv",
            "barley": "static/data/Barley.csv",
            "copra": "static/data/Copra.csv",
            "cotton": "static/data/Cotton.csv",
            "sesamum": "static/data/Sesamum.csv",
            "gram": "static/data/Gram.csv",
            "groundnut": "static/data/Groundnut.csv",
            "jowar": "static/data/Jowar.csv",
            "maize": "static/data/Maize.csv",
            "masoor": "static/data/Masoor.csv",
            "moong": "static/data/Moong.csv",
            "niger": "static/data/Niger.csv",
            "paddy": "static/data/Paddy.csv",
            "ragi": "static/data/Ragi.csv",
            "rape": "static/data/Rape.csv",
            "jute": "static/data/Jute.csv",
            "safflower": "static/data/Safflower.csv",
            "soyabean": "static/data/Soyabean.csv",
            "sugarcane": "static/data/Sugarcane.csv",
            "sunflower": "static/data/Sunflower.csv",
            "urad": "static/data/Urad.csv",
            'wheat': "static/data/Wheat.csv"
        }

        dataset = str(commodity_dict[cropsgraph])
        df = pd.read_csv(dataset)

        # EDA & Feature Engineering To Dataset

        df['Date'] = pd.to_datetime(df['Date'])
        df1 = df.drop(['Rainfall', 'Month', 'Year'], axis=1)
        Y = df["WPI"].to_numpy()
        X = df["Date"].to_numpy()
        df1.set_index('Date', inplace=True)

        # Using AD-Duckifuler test To make it Stationary

        df1['WPI First Difference'] = df1['WPI'] - df1['WPI'].shift(30)
        from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
        import statsmodels.api as sm
        from statsmodels.tsa.stattools import adfuller
        # adfuller_test(df1['WPI First Difference'].dropna())

        # importing SARIMAX For Time-Series Forecasting
        model = sm.tsa.statespace.SARIMAX(
            df1['WPI'], order=(1, 1, 1), seasonal_order=(1, 1, 1, 21))
        results = model.fit()

        # Creating Future 3 years dataset in order of months
        from pandas.tseries.offsets import DateOffset
        future_dates = [df1.index[-1] +
                        DateOffset(months=x)for x in range(0, 48)]
        future_datest_df = pd.DataFrame(
            index=future_dates[1:], columns=df1.columns)
        future_df = pd.concat([df1, future_datest_df])

        # Predicting the future values by SARIMAX Model
        future_df['forecast'] = results.predict(
            start=77, end=200, dynamic=True)
        # future_df[['WPI', 'forecast']].plot(figsize=(12, 8))

        new = future_df.iloc[80:, :]
        new_data = new.drop(['WPI', 'WPI First Difference'], axis=1)
        n1 = new_data.index.to_numpy()
        n2 = new_data["forecast"]
        n3 = new_data["forecast"]*basePrice[cropsgraph]
        n2.reset_index(drop=True).to_numpy()
        # Calculating Some Statistics Terms for output
        maxvalue = int(np.max(n2)*basePrice[cropsgraph])
        minvalue = int(np.min(n2)*basePrice[cropsgraph])
        avgvalue = int(np.average(n2)*basePrice[cropsgraph])

        # information abbout thr crop

        state = crop_data[cropsgraph][1]
        cropType = crop_data[cropsgraph][2]
        otherCountries = crop_data[cropsgraph][3]
        # Creating Dataframe for plotting Graphs
        df2 = pd.DataFrame({
            "Years(2018-2022)": n1,
            "Predicted WPI(Wholescale Price Index)": n2,

        })
        df8 = pd.DataFrame({
            "Years(2012-2019)": X,
            "WPI(Wholescale Price Index)": Y,

        })
        # USing PLotly to Plot the Graph

        fig = px.line(df2, x="Years(2018-2022)",
                      y="Predicted WPI(Wholescale Price Index)", title=f'Predicted Pice of {cropsgraph} in 2020-2023', markers=True)
        fig.update_traces(line_color='green')
        # fig.layout.plot_bgcolor='#fff'
        fig1 = px.line(df8, x="Years(2012-2019)",
                       y="WPI(Wholescale Price Index)", title=f'Precious Pice of {cropsgraph} in 2012-2019', markers=True, )
        graphJSON = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)
        graphJSON1 = json.dumps(fig1, cls=plotly.utils.PlotlyJSONEncoder)
        # deep=n2
        tableData = pd.DataFrame({
            "Years(2020-2023)": n1,
            "WPI(Wholescale Price Index)": n2,
            "Predicted Price(₹ )": (n2*basePrice[cropsgraph]).astype(int)

        })

        return render_template('forecast.html', graphJSON=graphJSON, graphJSON1=graphJSON1, max=maxvalue/100, min=minvalue/100, avg=avgvalue/100, cropanme=(cropsgraph).capitalize(), state=state, otherCountries=otherCountries, cropType=cropType.capitalize(), tables=[tableData.to_html(classes='styled-table', index=False)])
    else:

        return render_template("forecast.html")


# Selling PAge
@app.route('/sell', methods=['GET', 'POST'])
def sell():
    today = datetime.today()
    if request.method == 'POST':
        # Create variables for easy access
        _fname = request.form['fname']
        _lname = request.form['lname']
        _email = request.form['email']
        # password = request.form['password']
        _phone = request.form['phone']
        _add = request.form['add']
        _ctype = request.form['ctype']
        _quantity = request.form['Quantity']
        _cname = request.form['cname']
        _msp = request.form['msp']
        sellsave = Market(fname=_fname, lname=_lname, email=_email, phone=_phone, address=_add,
                        croptype=_ctype, quantity=_quantity, cropname=_cname, msp=_msp, registered_Date=today)
        sellsave.save()

        return render_template("sell.html")
    return render_template("sell.html")


# Plant disese prediction using Resenet9 (pretrained model)

disease_classes = ['Apple___Apple_scab',
                   'Apple___Black_rot',
                   'Apple___Cedar_apple_rust',
                   'Apple___healthy',
                   'Blueberry___healthy',
                   'Cherry_(including_sour)___Powdery_mildew',
                   'Cherry_(including_sour)___healthy',
                   'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot',
                   'Corn_(maize)___Common_rust_',
                   'Corn_(maize)___Northern_Leaf_Blight',
                   'Corn_(maize)___healthy',
                   'Grape___Black_rot',
                   'Grape___Esca_(Black_Measles)',
                   'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)',
                   'Grape___healthy',
                   'Orange___Haunglongbing_(Citrus_greening)',
                   'Peach___Bacterial_spot',
                   'Peach___healthy',
                   'Pepper,_bell___Bacterial_spot',
                   'Pepper,_bell___healthy',
                   'Potato___Early_blight',
                   'Potato___Late_blight',
                   'Potato___healthy',
                   'Raspberry___healthy',
                   'Soybean___healthy',
                   'Squash___Powdery_mildew',
                   'Strawberry___Leaf_scorch',
                   'Strawberry___healthy',
                   'Tomato___Bacterial_spot',
                   'Tomato___Early_blight',
                   'Tomato___Late_blight',
                   'Tomato___Leaf_Mold',
                   'Tomato___Septoria_leaf_spot',
                   'Tomato___Spider_mites Two-spotted_spider_mite',
                   'Tomato___Target_Spot',
                   'Tomato___Tomato_Yellow_Leaf_Curl_Virus',
                   'Tomato___Tomato_mosaic_virus',
                   'Tomato___healthy']

disease_model_path = './pickle/plant_disease_model.pth'
disease_model = ResNet9(3, len(disease_classes))
disease_model.load_state_dict(torch.load(
    disease_model_path, map_location=torch.device('cpu')))
disease_model.eval()


def predict_image(img, model=disease_model):
    """
    Transforms image to tensor and predicts disease label
    :params: image
    :return: prediction (string)
    """
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.ToTensor(),
    ])
    image = Image.open(io.BytesIO(img))
    img_t = transform(image)
    img_u = torch.unsqueeze(img_t, 0)

    # Get predictions from model
    yb = model(img_u)
    # Pick index with highest probability
    _, preds = torch.max(yb, dim=1)
    prediction = disease_classes[preds[0].item()]
    # Retrieve the class label
    return prediction

# Loading crop recommendation model


@app.route('/disease', methods=['GET', 'POST'])
def disease():
    title = 'Disease Detection'

    if request.method == 'POST':
        if 'file' not in request.files:
            return redirect(request.url)
        file = request.files.get('file')
        if not file:
            return render_template('disease.html', title=title)
        try:
            img = file.read()

            prediction = predict_image(img)

            prediction = Markup(str(disease_dic[prediction]))
            return render_template('disease.html', prediction=prediction, title=title)
        except:
            pass
    return render_template('disease.html', title=title)


@app.route('/news')
def news():
    from calendar import c


    headers = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/99.0.4844.74 Safari/537.36'}

    url = 'https://agriculturepost.com/category/farm-inputs/'
    r = requests.get(url, headers=headers)

    soup = BeautifulSoup(r.text, 'html.parser')
    print(soup.title.text)

    # news = soup.find_all('div', {'class': 'archive-content'})
    news=soup.find_all('article',{'class':'hitmag-post'})


    # print(len(news))
    newList = []
    for item in news:
        List = {
            'image':item.find('img')['src'],
            'title': item.find('h3', {'class': 'entry-title'}).text,
            'data': item.find('div', {'class': 'entry-summary'}).text,
            'date': item.find('time', {'class': 'entry-date'}).text,
            'content': item.find('p').text,
            'link': item.find('a')['href']
        }
        newList.append(List)

    return render_template('news.html',dict=newList)    


@app.route('/weather')
def weather():
    return render_template('weather.html')

if __name__ == "__main__":
    app.run(debug=True, port=5001)