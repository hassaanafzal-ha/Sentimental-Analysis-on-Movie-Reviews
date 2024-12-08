Sentiment Analysis for Movie Reviews

Overview
This project is a machine learning-based sentiment analysis tool that classifies movie reviews as either Positive or Negative. It features a graphical user interface (GUI) built using Python's tkinter, allowing users to interactively input reviews and select between two custom-trained machine learning models: Logistic Regression and Gradient Boosting.

Features
•	Custom implementation of Logistic Regression and Gradient Boosting algorithms.
•	Interactive GUI for model selection and review classification.
•	Custom SimpleVectorizer for text preprocessing and feature extraction.
•	Save and load trained models using pickle.

Project Structure
Project Folder/
│
├── gui.py                      # GUI application for sentiment analysis
├── code.ipynb                  # Notebook with model training and implementation
├── logistic_regression_model.pkl  # Pickled Logistic Regression model
├── gradient_boosting_model.pkl # Pickled Gradient Boosting model
├── vectorizer.pkl              # Pickled custom vectorizer
└── README.md                   # Project documentation

How to Run
Prerequisites
1.	Python 3.8 or later
2.	Required Python libraries: 
o	numpy
o	tkinter
o	pickle
o	scikit-learn (optional for debugging)

Install dependencies using:
pip install numpy

Steps
1.	Clone or download the project.
2.	Ensure the following files are in the same directory: 
o	gui.py
o	logistic_regression_model.pkl
o	gradient_boosting_model.pkl
o	vectorizer.pkl
3.	Run the GUI script: 
4.	python gui.py
   
Using the Application
1.	Launch the GUI by running gui.py.
2.	Select a model (Logistic Regression or Gradient Boosting) using the dropdown menu.
3.	Click Load Model to initialize the selected model.
4.	Enter a movie review in the text box.
5.	Click Classify to view the predicted sentiment.


Model Implementation
The machine learning models were coded manually without using pre-built libraries like scikit-learn for training:
•	Logistic Regression: Implements gradient descent for parameter optimization.
•	Gradient Boosting: Custom-coded boosting mechanism for decision trees.
All models were trained on a preprocessed dataset using the SimpleVectorizer class for feature extraction.

![image](https://github.com/user-attachments/assets/6a7d7127-cc4f-4729-9473-f7d5de734fb5)
![image](https://github.com/user-attachments/assets/c476ddc8-b224-4b14-902e-6264825f26d8)
![image](https://github.com/user-attachments/assets/f63d3cb3-e97a-4fa6-a342-ef35e8c4af33)
![image](https://github.com/user-attachments/assets/2c59d573-92f3-4859-b7ce-ac08b469f356)
![image](https://github.com/user-attachments/assets/622926db-8142-43b8-b4ce-1c53c189380c)






