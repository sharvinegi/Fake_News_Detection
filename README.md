# Fake News Detection using Flask and Machine Learning

This project is a web application that uses a machine learning model to detect fake news. The project leverages a **PassiveAggressiveClassifier** to predict whether a news article is fake or real, based on its content. The application also provides accuracy metrics and displays a confusion matrix for model performance.

## Features
- Web interface to input news article text and receive predictions.
- Visual representation of the confusion matrix for model evaluation.
- Accuracy score displayed on the prediction page.

## Project Structure
```bash
├── Fake_News_Det.py            # Main Flask application
├── model.pkl                   # Trained machine learning model
├── news.csv                    # Dataset used for training the model
├── templates/
│   └── index.html              # Frontend HTML template for user interaction
├── static/
│   └── confusion_matrix.png    # Generated confusion matrix image (after prediction)
├── README.md                   # Project documentation (this file)
```

## Requirements
Install the required libraries using:
```bash
pip install -r requirements.txt
```

### `requirements.txt`
```text
Flask
scikit-learn
pandas
matplotlib
seaborn
```

## Dataset
The `news.csv` file contains the news articles with labels indicating whether the news is fake or real. It is used to train the PassiveAggressiveClassifier.

## Running the Application
1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/fake-news-detector.git
   ```
2. Navigate into the project directory:
   ```bash
   cd fake-news-detector
   ```
3. Run the Flask application:
   ```bash
   python Fake_News_Det.py
   ```
4. Open your browser and go to `http://127.0.0.1:5000/` to use the web application.

## Usage
1. Enter a news article into the text box.
2. Click on "Predict" to receive a prediction on whether the news is **Real** or **Fake**.
3. The page will display the prediction along with the accuracy of the model and a confusion matrix visualization.

## Model Training
The model is trained using the **PassiveAggressiveClassifier** from `scikit-learn`. The training script is embedded within the Flask application, where the dataset (`news.csv`) is used for training and testing.

## Thanks
A special thanks to all the contributors and the open-source community that helped make this project possible. Your guidance, tools, and frameworks are greatly appreciated!

---

Feel free to use and modify this project. If you encounter any issues or have suggestions, don’t hesitate to reach out or submit a pull request. Thank you for checking out this project!
```
