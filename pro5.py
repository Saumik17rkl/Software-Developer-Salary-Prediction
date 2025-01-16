import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st

# Streamlit app title
st.title("Software Developer Salary Prediction")
st.write("This app predicts the salary of software developers based on country, education, and experience.")

# Define exchange rates for countries (example rates, replace with accurate ones)
exchange_rates = {
    'USA': 1.0,         # USD
    'UK': 0.75,         # GBP
    'France': 0.85,     # EUR
    'Brazil': 5.3,      # BRL
    'Germany': 0.85,    # EUR
    'Ukraine': 36.5,    # UAH
    'Canada': 1.3,      # CAD
    'Italy': 0.85,      # EUR
    'India': 82.0,      # INR
    'Spain': 0.85,      # EUR
    'Netherlands': 0.85, # EUR
    'Sweden': 10.5,     # SEK
    'Poland': 4.5,      # PLN
    'Australia': 1.5    # AUD
}

# Load and preprocess data
@st.cache_data
def load_and_preprocess_data():
    # Load data
    df = pd.read_csv('survey_results_public.csv')  # Updated file path
    df = df[['Country', 'EdLevel', 'YearsCodePro', 'Employment', 'ConvertedCompYearly']]
    df = df.rename(columns={'ConvertedCompYearly': 'Salary'})
    df = df[df['Salary'].notnull()]
    df = df.dropna()

    # Filter for full-time employment
    df = df[df['Employment'].str.contains('Employed', na=False)]
    df = df.drop('Employment', axis=1)

    # Shorten country categories
    def shorten_categories(categories, cutoff):
        categorical_map = {}
        for i in range(len(categories)):
            if categories.values[i] >= cutoff:
                categorical_map[categories.index[i]] = categories.index[i]
            else:
                categorical_map[categories.index[i]] = 'Other'
        return categorical_map

    country_map = shorten_categories(df['Country'].value_counts(), 400)
    df['Country'] = df['Country'].map(country_map)

    # Remove extreme salary values
    df = df[df['Salary'] <= 250000]
    df = df[df['Salary'] >= 10000]
    df = df[df['Country'] != 'Other']

    # Clean 'YearsCodePro' column
    def clean_experience(x):
        if x == 'More than 50 years':
            return 50
        if x == 'Less than 1 year':
            return 0.5
        if x == 'only bachelor\'s':
            return 0.0
        return float(x)

    df['YearsCodePro'] = df['YearsCodePro'].apply(clean_experience)

    # Clean 'EdLevel' column
    def clean_education(x):
        if "Bachelor's degree" in x:
            return "Bachelor's degree"
        if "Master's degree" in x:
            return "Master's degree"
        if "Professional degree" in x:
            return "Professional degree"
        return "Less than a Bachelor's"

    df['EdLevel'] = df['EdLevel'].apply(clean_education)

    # Manually encode categorical variables (Label Encoding)
    country_mapping = {country: idx for idx, country in enumerate(df['Country'].unique())}
    education_mapping = {education: idx for idx, education in enumerate(df['EdLevel'].unique())}

    df['Country'] = df['Country'].map(country_mapping)
    df['EdLevel'] = df['EdLevel'].map(education_mapping)

    return df, country_mapping, education_mapping


# Load data and preprocess
df, country_mapping, education_mapping = load_and_preprocess_data()

# Define allowed_countries based on the countries in the dataset
allowed_countries = list(country_mapping.keys())

# User Input Section
st.header("User Input Parameters")
country = st.selectbox("Select Country", allowed_countries)
education = st.selectbox("Select Education Level", list(education_mapping.keys()))

# Update Years of Experience as a scrollable dropdown
experience_options = ['Less than 1 year', 'More than 50 years', 'only bachelor\'s'] + list(range(1, 51))
experience = st.selectbox("Select Years of Experience", experience_options)

# Convert user input
experience_numeric = 0.5 if experience == 'Less than 1 year' else 50 if experience == 'More than 50 years' else 0.0 if experience == 'only bachelor\'s' else int(experience)

# Define features (X) and target (y)
X = df.drop("Salary", axis=1).values
y = df["Salary"].values

# Basic Decision Tree Regressor (Manually Implemented)
class DecisionTreeRegressorManual:
    def __init__(self, max_depth=10):
        self.max_depth = max_depth

    def fit(self, X, y):
        self.tree = self._build_tree(X, y, depth=0)

    def _build_tree(self, X, y, depth):
        if depth >= self.max_depth or len(set(y)) == 1:
            return np.mean(y)

        best_split = self._best_split(X, y)
        if best_split is None:
            return np.mean(y)

        left_mask = X[:, best_split['feature']] <= best_split['value']
        right_mask = ~left_mask

        left_tree = self._build_tree(X[left_mask], y[left_mask], depth + 1)
        right_tree = self._build_tree(X[right_mask], y[right_mask], depth + 1)

        return {'split': best_split, 'left': left_tree, 'right': right_tree}

    def _best_split(self, X, y):
        best_split = None
        best_score = float('inf')

        for feature in range(X.shape[1]):
            values = set(X[:, feature])
            for value in values:
                left_mask = X[:, feature] <= value
                right_mask = ~left_mask
                if np.any(left_mask) and np.any(right_mask):
                    left_y = y[left_mask]
                    right_y = y[right_mask]
                    score = self._score_split(left_y, right_y)
                    if score < best_score:
                        best_score = score
                        best_split = {'feature': feature, 'value': value}

        return best_split

    def _score_split(self, left_y, right_y):
        left_var = np.var(left_y)
        right_var = np.var(right_y)
        return left_var * len(left_y) + right_var * len(right_y)

    def predict(self, X):
        return np.array([self._predict_sample(x, self.tree) for x in X])

    def _predict_sample(self, x, tree):
        if isinstance(tree, dict):
            if x[tree['split']['feature']] <= tree['split']['value']:
                return self._predict_sample(x, tree['left'])
            else:
                return self._predict_sample(x, tree['right'])
        return tree

# Train model
model = DecisionTreeRegressorManual(max_depth=10)
model.fit(X, y)

# Prediction
if country not in country_mapping:
    st.error("Selected country not found in the dataset!")
else:
    input_country = country_mapping.get(country, -1)
    input_education = education_mapping.get(education, -1)

    if st.button("Predict Salary"):
        input_data = np.array([[input_country, input_education, experience_numeric]])
        predicted_salary_usd = model.predict(input_data)[0]

        country_name = list(country_mapping.keys())[list(country_mapping.values()).index(input_country)]
        exchange_rate = exchange_rates.get(country_name, 1.0)
        predicted_salary_local = predicted_salary_usd * exchange_rate

        st.subheader(f"Predicted Salary in USD: ${predicted_salary_usd:,.2f}")
        st.subheader(f"Predicted Salary in {country_name} currency: {predicted_salary_local:,.2f}")

# Visualize data
if st.checkbox("Show Data Visualization"):
    st.write("### Salary Distribution")
    plt.figure(figsize=(10, 5))
    sns.histplot(df['Salary'], bins=30, kde=True, color='blue')
    st.pyplot(plt)

    st.write("### Salary vs. Years of Experience")
    plt.figure(figsize=(10, 5))
    sns.scatterplot(x='YearsCodePro', y='Salary', data=df, color='green')
    st.pyplot(plt)

# Model Evaluation
if st.checkbox("Show Model Evaluation Metrics"):
    mse = np.mean((y - model.predict(X)) ** 2)
    st.write(f"MSE: {mse:.2f}")
