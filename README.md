# ❤️ Heart Disease Prediction Using Machine Learning

[![Python](https://img.shields.io/badge/Python-3.x-blue.svg)](https://python.org)
[![Scikit-learn](https://img.shields.io/badge/Scikit--learn-Latest-orange.svg)](https://scikit-learn.org)
[![Pandas](https://img.shields.io/badge/Pandas-Latest-lightgrey.svg)](https://pandas.pydata.org)
[![NumPy](https://img.shields.io/badge/NumPy-Latest-lightblue.svg)](https://numpy.org)

A machine learning model to predict heart disease using logistic regression. This project analyzes medical indicators to assist in early detection of potential heart conditions.

## 🎯 Performance Metrics

- **Training Accuracy**: 85.24%
- **Test Accuracy**: 80.49%
- **Model Status**: Good generalization with minimal overfitting
- **Prediction Type**: Binary classification (0: No Heart Disease, 1: Heart Disease)

## 📊 Dataset

- **Source**: [Kaggle Heart Disease Dataset](https://www.kaggle.com/datasets/johnsmith88/heart-disease-dataset)
- **Records**: 1025 entries
- **Features**: 14 (13 input features + 1 target)
- **Distribution**: 
  - Healthy (0): 499 cases
  - Heart Disease (1): 526 cases


    <table class="feature-table">
        <thead>
            <tr>
                <th>Feature</th>
                <th>Description</th>
                <th>Range/Type</th>
            </tr>
        </thead>
        <tbody>
            <tr>
                <td>age</td>
                <td>Age of the patient</td>
                <td>29-77 years</td>
            </tr>
            <tr>
                <td>sex</td>
                <td>Gender</td>
                <td>0: Female, 1: Male</td>
            </tr>
            <tr>
                <td>cp</td>
                <td>Chest pain type</td>
                <td>0-3</td>
            </tr>
            <tr>
                <td>trestbps</td>
                <td>Resting blood pressure</td>
                <td>94-200 mm Hg</td>
            </tr>
            <tr>
                <td>chol</td>
                <td>Serum cholesterol</td>
                <td>126-564 mg/dl</td>
            </tr>
            <tr>
                <td>fbs</td>
                <td>Fasting blood sugar > 120 mg/dl</td>
                <td>0: False, 1: True</td>
            </tr>
            <tr>
                <td>restecg</td>
                <td>Resting ECG results</td>
                <td>0-2</td>
            </tr>
            <tr>
                <td>thalach</td>
                <td>Maximum heart rate achieved</td>
                <td>71-202</td>
            </tr>
            <tr>
                <td>exang</td>
                <td>Exercise induced angina</td>
                <td>0: No, 1: Yes</td>
            </tr>
            <tr>
                <td>oldpeak</td>
                <td>ST depression induced by exercise</td>
                <td>0-6.2</td>
            </tr>
            <tr>
                <td>slope</td>
                <td>Slope of peak exercise ST segment</td>
                <td>0-2</td>
            </tr>
            <tr>
                <td>ca</td>
                <td>Number of major vessels colored by fluoroscopy</td>
                <td>0-4</td>
            </tr>
            <tr>
                <td>thal</td>
                <td>Thalassemia</td>
                <td>0-3</td>
            </tr>
        </tbody>
    </table>


## 🛠️ Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/heart-disease-prediction.git
cd heart-disease-prediction
```

2. Create and activate virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install required packages:
```bash
pip install -r requirements.txt
```

## 💻 Usage

### Load and Prepare Data
```python
import pandas as pd
from sklearn.model_selection import train_test_split

# Load data
heart_data = pd.read_csv('heart.csv')

# Split features and target
X = heart_data.drop(columns='target', axis=1)
Y = heart_data['target']

# Split into training and test sets
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, stratify=Y, random_state=2)
```

### Train Model
```python
from sklearn.linear_model import LogisticRegression

# Initialize and train model
model = LogisticRegression()
model.fit(X_train, Y_train)
```

### Make Predictions
```python
def predict_heart_disease(input_data):
    # Example input: (43, 0, 0, 132, 341, 1, 0, 136, 1, 3, 1, 0, 3)
    input_array = np.asarray(input_data).reshape(1, -1)
    prediction = model.predict(input_array)
    return "Heart Disease Detected" if prediction[0] == 1 else "No Heart Disease Detected"
```

## 📈 Results

```python
# Model Evaluation
print(f'Training Accuracy: {accuracy_score(model.predict(X_train), Y_train):.2%}')
print(f'Testing Accuracy: {accuracy_score(model.predict(X_test), Y_test):.2%}')
```

## 🔄 Future Improvements

- [ ] Implement feature scaling
- [ ] Add cross-validation
- [ ] Try different algorithms (Random Forest, SVM)
- [ ] Add feature importance analysis
- [ ] Include ROC curve analysis
- [ ] Add confusion matrix visualization
- [ ] Perform hyperparameter tuning

## 🤝 Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- Dataset provided by [Kaggle](https://www.kaggle.com/datasets/johnsmith88/heart-disease-dataset)
- Inspiration from various heart disease research papers
- scikit-learn documentation and community

## 📧 Contact

My Name - Vislavath Pavani

Project Link: [(https://mybinder.org/v2/gh/12pavani/Heart-Disease-Prediction-Model.git/main)](https://mybinder.org/v2/gh/12pavani/Heart-Disease-Prediction-Model/69879853fe0cf7be0cad78b563af53f007784901?urlpath=lab%2Ftree%2FHeart%20Disease%20Prediction.ipynb)](https://mybinder.org/v2/gh/12pavani/Heart-Disease-Prediction-Model/69879853fe0cf7be0cad78b563af53f007784901?urlpath=lab%2Ftree%2FHeart%20Disease%20Prediction.ipynb)
