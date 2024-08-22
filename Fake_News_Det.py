from flask import Flask, render_template, request
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
import pickle
import pandas as pd
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns

app = Flask(__name__)
tfvect = TfidfVectorizer(stop_words='english', max_df=0.7)
loaded_model = pickle.load(open('model.pkl', 'rb'))
dataframe = pd.read_csv('news.csv')
x = dataframe['text']
y = dataframe['label']
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)

def fake_news_det(news):
    tfid_x_train = tfvect.fit_transform(x_train)
    tfid_x_test = tfvect.transform(x_test)
    input_data = [news]
    vectorized_input_data = tfvect.transform(input_data)
    prediction = loaded_model.predict(vectorized_input_data)
    accuracy, confusion_matrix_image_path = calculate_metrics()
    return prediction, accuracy, confusion_matrix_image_path
def calculate_metrics():
    tfid_x_test = tfvect.transform(x_test)
    y_pred = loaded_model.predict(tfid_x_test)
    acc = accuracy_score(y_test, y_pred)
    
   
    cm = confusion_matrix(y_test, y_pred)
    
 
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Real', 'Fake'], yticklabels=['Real', 'Fake'])
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')
    plt.tight_layout()
    
   
    confusion_matrix_image_path = 'static/confusion_matrix.png'
    plt.savefig(confusion_matrix_image_path)
    
    return acc, confusion_matrix_image_path

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        message = request.form['message']
        pred, accuracy, confusion_matrix_image_path = fake_news_det(message)
        print("Prediction:", pred)
        print("Accuracy:", accuracy)
        return render_template('index.html', prediction=pred, accuracy=accuracy, plot_url=confusion_matrix_image_path)
    else:
        return render_template('index.html', prediction="Something went wrong")
    
if __name__ == '__main__':
    app.run(debug=True)