# Software-Developer-Salary-Prediction

This project is a Streamlit web application that predicts the salary of software developers based on their country, education level, and years of professional experience. It leverages Decision Tree Regression to predict the salary in USD, with an option to view the predicted salary in the local currency using exchange rates. The application also provides data visualizations to explore the dataset, showcasing salary distribution and the relationship between experience and salary.

Key Features:
User Inputs:

Select country, education level, and years of experience.
The application uses a user-friendly interface for easy parameter selection.
Salary Prediction:

Predicts the salary of a software developer based on the provided input.
Shows the salary prediction in both USD and the local currency (using exchange rates).
Data Visualization:

Displays a salary distribution histogram and a scatter plot of salary vs. years of experience.
Model Implementation:

The model is a Decision Tree Regressor that is trained on a dataset of software developers' salaries, with features including country, education level, and years of professional experience.
Technologies Used:
Streamlit: For creating the interactive web app.
Pandas: For data manipulation and preprocessing.
NumPy: For numerical operations.
Matplotlib & Seaborn: For data visualization.
Decision Tree Regressor: For salary prediction.
Dataset:
The dataset used in this project is derived from the Stack Overflow Developer Survey, which includes information on software developers' demographics, education, experience, and salary.

How to Run:
Clone the repository:
bash
Copy
Edit
git clone https://github.com/Saumik17rkl/Software-Developer-Salary-Prediction.git
Install the required libraries:
bash
Copy
Edit
pip install -r requirements.txt
Run the Streamlit app:
bash
Copy
Edit
streamlit run app.py
Screenshots:
Home Page: The user interface where parameters are input for salary prediction.
Prediction Result: Displaying the predicted salary in USD and local currency.
Data Visualizations: Interactive charts showing salary distribution and salary vs. years of experience.
Contributions:
Feel free to fork the repository, contribute to improvements, and submit issues or pull requests.

This description provides a comprehensive overview of the app and its features. You can modify or extend it based on additional details you want to share.
