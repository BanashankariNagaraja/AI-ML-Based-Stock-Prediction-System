from flask import Flask, render_template, request, redirect, session, flash, send_file
from flask_sqlalchemy import SQLAlchemy
from werkzeug.security import generate_password_hash, check_password_hash
import pandas as pd
import yfinance as yf
from sklearn.linear_model import LinearRegression
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os
import sqlite3
from utils.prediction import predict_stock

app = Flask(__name__)

app.secret_key = "stock_secret_key"

# =========================
# DATABASE CONFIGURATION
# =========================

app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///database.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

db = SQLAlchemy(app)

# =========================
# DATABASE MODEL
# =========================

class User(db.Model):

    id = db.Column(db.Integer, primary_key=True)

    username = db.Column(db.String(100), nullable=False)

    email = db.Column(db.String(100), unique=True, nullable=False)

    password = db.Column(db.String(200), nullable=False)

    role = db.Column(db.String(20), default='user')

class Message(db.Model):

    id = db.Column(db.Integer, primary_key=True)

    name = db.Column(db.String(100))

    email = db.Column(db.String(100))

    subject = db.Column(db.String(200))

    message = db.Column(db.Text)
# =========================
# HOME PAGE
# =========================

@app.route('/')
def home():

    return render_template('index.html')

# =========================
# ABOUT PAGE
# =========================

@app.route('/about')
def about():

    return render_template('about.html')

# =========================
# CONTACT PAGE
# =========================

@app.route('/contact', methods=['GET', 'POST'])
def contact():

    if request.method == 'POST':

        try:

            name = request.form['name']

            email = request.form['email']

            subject = request.form['subject']

            message_text = request.form['message']

            new_message = Message(

                name=name,

                email=email,

                subject=subject,

                message=message_text

            )

            db.session.add(new_message)

            db.session.commit()

            flash('Message Sent Successfully')

            return redirect('/contact')

        except Exception as e:

            flash(f'Error: {e}')

    return render_template('contact.html')
# =========================
# REGISTER
# =========================

@app.route('/register', methods=['GET', 'POST'])
def register():

    if request.method == 'POST':

        username = request.form['username']

        email = request.form['email']

        password = request.form['password']

        existing_user = User.query.filter_by(
            email=email
        ).first()

        if existing_user:

            flash("Email already exists")

            return redirect('/register')

        hashed_password = generate_password_hash(
            password
        )

        new_user = User(

            username=username,

            email=email,

            password=hashed_password

        )

        db.session.add(new_user)

        db.session.commit()

        flash("Registration Successful")

        return redirect('/login')

    return render_template('register.html')

# =========================
# LOGIN
# =========================

@app.route('/login', methods=['GET', 'POST'])
def login():

    if request.method == 'POST':

        email = request.form['email']

        password = request.form['password']

        user = User.query.filter_by(
            email=email
        ).first()

        if user and check_password_hash(
            user.password,
            password
        ):

            session['user_id'] = user.id

            session['username'] = user.username

            session['role'] = user.role

            flash("Login Successful")

            if user.role == 'admin':

                return redirect('/admin')

            return redirect('/dashboard')

        else:

            flash("Invalid Email or Password")

    return render_template('login.html')

# =========================
# LOGOUT
# =========================

@app.route('/logout')
def logout():

    session.clear()

    flash("Logged Out Successfully")

    return redirect('/')

# =========================
# USER DASHBOARD
# =========================

@app.route('/dashboard')
def dashboard():

    if 'user_id' not in session:

        flash("Please Login First")

        return redirect('/login')

    return render_template('dashboard.html')

# =========================
# PROFILE
# =========================

@app.route('/profile', methods=['GET', 'POST'])
def profile():

    if 'user_id' not in session:

        return redirect('/login')

    user = User.query.get(
        session['user_id']
    )

    if request.method == 'POST':

        user.username = request.form['username']

        user.email = request.form['email']

        db.session.commit()

        flash("Profile Updated Successfully")

    return render_template(
        'profile.html',
        user=user
    )

# =========================
# DELETE PROFILE
# =========================

@app.route('/delete_profile')
def delete_profile():

    if 'user_id' not in session:

        return redirect('/login')

    user = User.query.get(
        session['user_id']
    )

    db.session.delete(user)

    db.session.commit()

    session.clear()

    flash("Profile Deleted")

    return redirect('/')

# =========================
# REALTIME STOCK PRICE
# =========================

@app.route('/realtime', methods=['GET', 'POST'])
def realtime():

    price_usd = None
    price_inr = None
    ticker = None

    if request.method == 'POST':

        ticker = request.form['ticker'].upper()

        stock = yf.Ticker(ticker)

        data = stock.history(period='1d')

        if not data.empty:

            # USD PRICE

            price_usd = round(
                float(data['Close'].iloc[-1]),
                2
            )

            # USD TO INR CONVERSION

            usd_to_inr = 83.50

            price_inr = round(
                price_usd * usd_to_inr,
                2
            )

    return render_template(

        'realtime.html',

        price_usd=price_usd,

        price_inr=price_inr,

        ticker=ticker

    )
# =========================
# STOCK PREDICTION
# =========================

@app.route('/predict', methods=['GET', 'POST'])
def predict():

    prediction = None
    prediction_inr = None
    chart = None
    ticker_name = None
    csv_path = None

    if request.method == 'POST':

        try:

            ticker = request.form['ticker'].upper()

            ticker_name = ticker

            # DOWNLOAD DATA

            df = yf.download(
                ticker,
                period='6mo',
                auto_adjust=True
            )

            if df.empty:

                flash("Invalid Stock Symbol")

                return render_template(
                    'predict.html',
                    prediction=None
                )

            # SAVE DATASET CSV

            csv_path = f"static/charts/{ticker}_data.csv"

            df.to_csv(csv_path)

            # STOCK DATA

            df = df[['Close']].copy()

            # CREATE PREDICTION COLUMN

            df['Prediction'] = df['Close'].shift(-2)

            # REMOVE NULL VALUES

            df = df.dropna()

            # X AND Y

            X = np.array(df[['Close']])

            y = np.array(df['Prediction'])

            # TRAIN MODEL

            model = LinearRegression()

            model.fit(X, y)

            # LAST 2 DAYS

            x_future = np.array(
                df[['Close']].tail(2)
            )

            # PREDICTION

            prediction_values = model.predict(
                x_future
            )

            # USD PREDICTION

            prediction = [

                round(float(prediction_values[0]), 2),

                round(float(prediction_values[1]), 2)

            ]

            # INR CONVERSION

            usd_to_inr = 83.50

            prediction_inr = [

                round(prediction[0] * usd_to_inr, 2),

                round(prediction[1] * usd_to_inr, 2)

            ]

            # CREATE GRAPH

            plt.figure(figsize=(10,5))

            plt.plot(
                df['Close'],
                label='Closing Price',
                linewidth=2
            )

            plt.title(
                f'{ticker} Stock Price'
            )

            plt.xlabel("Days")

            plt.ylabel("Price")

            plt.legend()

            plt.grid(True)

            # SAVE GRAPH

            chart_path = f"static/charts/{ticker}.png"

            plt.savefig(
                chart_path,
                bbox_inches='tight'
            )

            plt.close()

            chart = chart_path

            # RETURN TEMPLATE

            return render_template(

                'predict.html',

                prediction=prediction,

                prediction_inr=prediction_inr,

                chart=chart,

                ticker=ticker_name,

                csv_path=csv_path

            )

        except Exception as e:

            flash(f"Prediction Error : {e}")

            return render_template(
                'predict.html',
                prediction=None
            )

    return render_template(
        'predict.html',
        prediction=None
    )
# =========================
# EDUCATION PAGE
# =========================

@app.route('/education')
def education():

    return render_template(
        'education.html'
    )

# =========================
# DOWNLOAD STOCK TICKERS
# =========================

@app.route('/download')
def download():

    data = {

        'Company': [

            'Apple',
            'Tesla',
            'Google',
            'Microsoft',
            'Amazon'

        ],

        'Ticker': [

            'AAPL',
            'TSLA',
            'GOOG',
            'MSFT',
            'AMZN'

        ]
    }

    df = pd.DataFrame(data)

    file_path = "tickers.csv"

    df.to_csv(file_path, index=False)

    return send_file(

        file_path,

        as_attachment=True

    )

# =========================
# ADMIN DASHBOARD
# =========================

@app.route('/admin')
def admin_dashboard():

    if session.get('role') != 'admin':

        flash("Admin Access Only")

        return redirect('/login')

    total_users = User.query.count()

    return render_template(

        'admin/admin_dashboard.html',

        total_users=total_users

    )
@app.route('/admin/messages')
def admin_messages():

    messages = Message.query.order_by(
        Message.id.desc()
    ).all()

    return render_template(
        'admin/messages.html',
        messages=messages
    )

# =========================
# ANALYTICS PAGE
# =========================

@app.route('/analytics')
def analytics():

    if session.get('role') != 'admin':

        return redirect('/login')

    total_users = User.query.count()

    return render_template(

        'admin/analytics.html',

        total_users=total_users

    )

# =========================
# MANAGE USERS
# =========================

@app.route('/admin/users')
def manage_users():

    if session.get('role') != 'admin':

        return redirect('/login')

    users = User.query.all()

    return render_template(

        'admin/users.html',

        users=users

    )

# =========================
# DELETE USER
# =========================

@app.route('/admin/delete_user/<int:id>')
def delete_user(id):

    if session.get('role') != 'admin':

        return redirect('/login')

    user = User.query.get(id)

    if user:

        db.session.delete(user)

        db.session.commit()

        flash("User Deleted Successfully")

    return redirect('/admin/users')

# =========================
# MAIN
# =========================

if __name__ == '__main__':

    # CREATE CHARTS FOLDER

    if not os.path.exists('static/charts'):

        os.makedirs('static/charts')

    # CREATE DATABASE

    with app.app_context():

        db.create_all()

        # DEFAULT ADMIN

        admin = User.query.filter_by(
            email='admin@gmail.com'
        ).first()

        if not admin:

            admin_user = User(

                username='Admin',

                email='admin@gmail.com',

                password=generate_password_hash(
                    'admin123'
                ),

                role='admin'

            )

            db.session.add(admin_user)

            db.session.commit()

    app.run(debug=True)