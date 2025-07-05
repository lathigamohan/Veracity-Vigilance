# 🧠 Veracity Vigilance: Detecting Fake News with Machine Learning

## 📅 Overview

"Veracity Vigilance" is a machine learning-based fake news detection system designed to differentiate between real and fake news articles using NLP and classification techniques. In an age of misinformation, this project aims to support truth-checking by intelligently analyzing news content for authenticity.

---

## 📂 Datasets Used

### 1. **Kaggle - Fake and Real News Dataset**

* Source: [Kaggle - Clément Bisaillon](https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset)
* Format: Two CSVs - `Fake.csv` and `True.csv`
* Columns: `title`, `text`, `subject`, `date`
* Labels:

  * `0` for Fake News (from Fake.csv)
  * `1` for Real News (from True.csv)

### 2. **Jaipoona Dataset (Custom / Experimental)**

* Used in: `project_code.ipynb`
* Description: A custom-curated or academic dataset containing simplified, shorter news statements, often headline-style.
* Columns: `text`, `label`
* Purpose: Used to experiment with alternate text styles (e.g., social media format), improving the model’s robustness to different types of news.

### 📊 Dataset Comparison (Summary)

| Feature          | Kaggle Dataset                  | Jaipoona Dataset           |
| ---------------- | ------------------------------- | -------------------------- |
| Text Length      | Long articles with full context | Short headlines/statements |
| Source Diversity | Multiple publications/newsrooms | Mixed/unclear sources      |
| Format           | Full CSVs with metadata         | Cleaned text-label pairs   |
| Use Case         | Core model training             | Extended experimentation   |

---

## 💪 Technologies Used

* **Language**: Python 3.x
* **Text Preprocessing**: NLTK, TextBlob, Regex
* **Vectorization**: TfidfVectorizer (Sklearn)
* **ML Models**:

  * Logistic Regression
  * Random Forest
  * Support Vector Machine (SVM)
  * XGBoost Classifier
  * Voting Ensemble (Soft Voting)
* **Model Evaluation**:

  * Accuracy, Precision, Recall, F1 Score
  * Confusion Matrix
  * Classification Report
  * ROC Curve, Feature Importance
* **Visualization**:

  * Seaborn, Matplotlib, WordCloud
  * Plotly (for interactive visualizations)

---

## 🔬 Key Features

*  **Real-time Prediction**: User can input a news article and instantly get a prediction (Real or Fake)
*  **Confidence Score Bar Chart**: Visualization of model probability (e.g., 92% Real, 8% Fake)
*  **Feature Importance**: Top influential words shown via bar chart (XGBoost)
*  **Visual Analysis**: Includes EDA plots, word clouds, class distribution histograms

---

## 📊 Model Performance Snapshot

| Model               | Accuracy |
| ------------------- | -------- |
| Logistic Regression | 94%      |
| Random Forest       | 93%      |
| XGBoost             | 95%      |
| Voting Classifier   |  96%   |

---

## 📚 How to Use

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Run the notebook
project_code.ipynb

# 3. Paste an article in the input field
# 4. View prediction and probability chart
```

---

## 🚀 Future Enhancements

* Add Transformer models (e.g., BERT)
* Streamlit-based UI for web use
* Add support for multilingual news detection
* Deploy to HuggingFace or Streamlit Cloud

---

## 👤 Author

**Lathiga** – Machine learning enthusiast & NLP explorer.
This project is a step toward building trustworthy AI for social good.

---

## 🤝 Connect With Me

*  Email: [your-email@example.com](lathiga1207@gmail.com)
*  LinkedIn: [linkedin.com/in/your-profile](linkedin.com/in/lathiga-416aa6321)
*  GitHub: [github.com/your-github](github.com/lathigamohan)

Let's connect and collaborate on more projects! 🚀
