from flask import Flask , render_template, request
from spam_mail import load_mail_data , train_spam_model,predict_spam

app = Flask(__name__)

data = load_mail_data()
model , features_extraction , train_acc , test_acc = train_spam_model(data)

@app.route('/')
def home():
    return render_template("index.html")

@app.route('/predict', methods=['POST'])
def predict():
    try:
        input_mail = request.form['message']
        preduction = predict_spam(model , features_extraction, input_mail)
    except Exception as e:
        prediction = f"⚠️ Error: {str(e)}"
    return render_template("index.html", prediction=prediction, message=input_mail,
                           train_accuracy=train_acc, test_accuracy=test_acc)

if __name__ == '__main__':
    app.run(debug=True)
