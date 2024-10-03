
from flask import Flask, render_template, request
from URL_model import predict_url
from Text_model import predict_text

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')


@app.route('/detect', methods=['POST'])
def detect():
    # Get user input from the form
    text = request.form['text']
    url = request.form['url']

    # Call functions from detection Python files
    text_outcome = predict_text(text)
    url_outcome = predict_url(url)

    # Render template with outcomes
    return render_template('index.html', text_outcome=text_outcome, url_outcome=url_outcome)

if __name__ == '__main__':
    app.run(debug=True)
