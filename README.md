# 🍽️ Restaurant Review Sentiment Analysis

## 📖 Introduction
This project focuses on analyzing public sentiment from restaurant reviews, sourced from Kaggle. The dataset consists of customer opinions, star ratings, and timestamps, offering insights into dining experiences across numerous restaurants.

## 🎯 Objective
The main goal is to gain insights into customer sentiment trends and build models that can accurately classify review sentiment into positive, neutral, or negative categories.

## 🧰 Models Used
Four models are tested for sentiment classification:

1. Logistric Regression
2. Naive Bayes
3. Support Vector Machine (SVM)
4. Distiled Bidirectional Encoder Representations from Transformers (DistilBERT)
   
## 📊 Result Highlights
- Data Preprocessing: Included sentiment data cleaning and augmentation using back-translation to address class imbalance.
- Traditional Models (Logistic Regression, Naive Bayes, Linear SVC):
   - Achieved F1-scores above 0.78 (weighted average) after tuning.
   - Struggled most with the Neutral class, which showed limited improvement even after adjustments.
- DistilBERT (Base Model):
   - Achieved an overall accuracy of 83% and a macro F1-score of 0.77, with strong performance on Positive (F1=0.91) and Negative (F1=0.84) classes, but lower on Neutral (F1=0.56).
- DistilBERT with Class Weights:
   - Improved Neutral class F1-score to 0.61, maintaining high accuracy (81%) and macro F1-score (0.78). Class balancing helped increase recall for the minority class.

## 🔮 Future Work
- Test larger Transformer models like BERT and RoBERTa for better performance
- Use more data augmentation and include external sentiment data to enrich the dataset
- Apply explainable AI methods to better understand model decisions

