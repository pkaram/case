from flask import Flask, render_template, request
from utils import InputForm, search_similar_text

app = Flask(__name__, template_folder='templates')


@app.route("/", methods=['GET', 'POST'])
def index():
    form = InputForm(request.form)
    if request.method == 'POST' and form.validate():        
        text = request.form['Text']
        similar_text_docs = search_similar_text(text)
    else:
        similar_text_docs = None

    return render_template('index.html',
    form=form,
    result=similar_text_docs
    )

if __name__ == '__main__':
    app.run(port=5000,debug=True)