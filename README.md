# Ethiopia_food_decisiontree

# 🌾 Ethiopian Food Market Analysis using Decision Tree Classifier  

This repository contains a machine learning project analyzing the Ethiopian food market using the Decision Tree Classifier. The study explores pricing volatility, supply chain challenges, and market dynamics, achieving 96% accuracy in predictions.

## 📌 Overview  
This project analyzes the Ethiopian food market dataset to uncover key insights into pricing, distribution, and market trends. The **Decision Tree Classifier** was applied to predict market patterns with high accuracy, helping identify disparities and improve strategic decision-making.  

## 🎯 Objectives  
- Apply **Decision Tree Classifier** to predict Ethiopian food market trends.  
- Identify significant patterns in product pricing and distribution.  
- Provide insights for stakeholders to address **supply chain and pricing volatility**.  
- Support better decision-making for **food security and sustainability**.

## 🧑‍💻 Methodology  
- **Dataset:** Ethiopia Weekly FEWS NET Staple Food Price Data (90,558 rows, 15 columns).  
- **Preprocessing Steps:**  
  - Missing values (25% in "Value") filled with median.  
  - One-hot encoding applied to categorical variables.  
  - Outliers dected but were not removed if anyone wants to remove the outliers it can be removed using z-scores.  
- **Algorithm Used:** Decision Tree Classifier.  
- **Evaluation Metrics:** Accuracy, Sensitivity, Specificity, F1 Score.

## 📊 Results  

### Model Accuracy  
The Decision Tree Classifier achieved **96.34% accuracy** with perfect sensitivity, specificity, and F1-score across folds.  

<img width="478" height="137" alt="image" src="https://github.com/user-attachments/assets/a192e74e-cd65-4e66-b993-640e52d69db2" />

### Market Insights  
Analysis revealed pricing disparities in different Ethiopian markets:  
- Essential imported goods (rice, sugar, oil) showed **low quantities**.  
- Beddenno market had consistently **higher costs** for basic items like diesel, oil, and whole grain.

<img width="483" height="285" alt="image" src="https://github.com/user-attachments/assets/a6dee966-6114-4462-b3c4-6b703d3aef4e" />

<img width="515" height="266" alt="image" src="https://github.com/user-attachments/assets/6ecb5ee3-c5ea-4f4a-b853-21558ed09745" />


## 📂 Repository Structure  

├── data/ # Raw and cleaned datasets

├── notebooks/ # Jupyter notebooks for Decision Tree analysis

├── src/ # Python scripts (preprocessing, training, evaluation)

├── images/ # Visuals used in README

├── results/ # Model outputs and performance reports

└── README.md # Project documentation



## Table 3 – Model Comparison

To evaluate the robustness of the Decision Tree Classifier, multiple machine learning algorithms were compared. Below Table summarizes the performance metrics (Accuracy, Precision, Recall, F1-Score) across models.
- The Decision Tree Classifier outperformed others with 96.34% accuracy, making it the most reliable choice for this dataset.
- Logistic Regression and KNN showed moderate performance but struggled with class imbalance.
- Random Forest performed competitively but at higher computational cost.

<img width="388" height="392" alt="image" src="https://github.com/user-attachments/assets/e9432875-78f8-49e3-aa2d-ad7bd4c75992" />

## 🗺️ Fig. 8 – Regional Price Trends

- Regional analysis highlighted disparities across Ethiopian markets.
- Beddenno consistently showed higher average costs for staples like diesel and grain.
- Central markets demonstrated more stable pricing, while peripheral regions faced volatility.
- These findings suggest a need for targeted regional strategies to address supply chain inefficiencies.

<img width="329" height="190" alt="image" src="https://github.com/user-attachments/assets/27422151-8501-42c7-9be6-bbd6ecd704fd" />

## Conclusion:

This study demonstrates the value of machine learning in food market analysis, providing actionable insights for stakeholders in Ethiopia’s agricultural and food supply sectors. By identifying key market drivers and regional disparities, the model supports:
- Policy decisions to reduce supply chain inefficiencies.
- Pricing strategies that mitigate volatility.
- Food security planning that prioritizes vulnerable regions.
- Ultimately, the Decision Tree Classifier proved to be not only accurate but also transparent, making it a strong choice for real-world decision-making in emerging markets.

## 🔮 Future Scope
- Extend Decision Tree analysis with more market-specific features.
- Explore time-series forecasting for price volatility.
- Apply insights to food security policies and regional planning.








