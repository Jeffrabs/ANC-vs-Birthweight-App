
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# ---------------------------------
# 1. PAGE CONFIGURATION
# ---------------------------------
st.set_page_config(
    page_title="ANC and Birth Weight Research",
    page_icon="👶",
    layout="wide"
)

st.title("Effect of Antenatal Care Visits on Birth Weight")
st.subheader("A Machine Learning Research Application")

st.write(
    "This application explores the relationship between the number of "
    "antenatal care (ANC) visits and baby birth weight using Linear Regression."
)

st.info(
    "Research demonstration: predictions are statistical estimates, "
    "not medical diagnoses or proof that ANC visits cause changes in birth weight."
)

# ---------------------------------
# 2. LOAD CSV DATASET
# ---------------------------------
st.sidebar.header("Research Navigation")

page = st.sidebar.radio(
    "Select a page",
    [
        "Home",
        "About the Research",
        "Dataset",
        "Data Analysis",
        "Birth Weight Prediction",
        "Model Evaluation"
    ]
)

csv_files = list(Path(".").glob("*.csv"))

if not csv_files:
    st.error(
        "No CSV dataset was found. Upload your cleaned CSV file "
        "to the same GitHub repository as app.py."
    )
    st.stop()

if len(csv_files) == 1:
    selected_file = csv_files[0]
else:
    selected_file = st.sidebar.selectbox(
        "Select your dataset",
        csv_files,
        format_func=lambda x: x.name
    )

try:
    df = pd.read_csv(selected_file)
except Exception as e:
    st.error(f"Could not read the dataset: {e}")
    st.stop()

st.sidebar.caption(f"Dataset: {selected_file.name}")

# ---------------------------------
# 3. SELECT DATA COLUMNS
# ---------------------------------
st.sidebar.header("Select Dataset Columns")

st.write("### Dataset column configuration")

columns = df.columns.tolist()

def find_column(possible_names):
    for name in possible_names:
        for col in columns:
            if col.strip().lower() == name.lower():
                return col
    return columns[0]

default_anc = find_column(["ANC Visits", "ANC Visit", "Number of ANC Visits"])
default_weight = find_column(["Birth Weight", "Birth Weight in Grams", "Birth Weight (g)"])

anc_col = st.sidebar.selectbox(
    "ANC visits column",
    columns,
    index=columns.index(default_anc)
)

weight_col = st.sidebar.selectbox(
    "Birth weight column",
    columns,
    index=columns.index(default_weight)
)

if anc_col == weight_col:
    st.error("Please select two different columns.")
    st.stop()

# ---------------------------------
# 4. CLEAN DATA FOR MODEL
# ---------------------------------
model_df = df[[anc_col, weight_col]].copy()

model_df[anc_col] = pd.to_numeric(
    model_df[anc_col], errors="coerce"
)

model_df[weight_col] = pd.to_numeric(
    model_df[weight_col], errors="coerce"
)

model_df = model_df.dropna()

model_df = model_df[
    (model_df[anc_col] >= 0) &
    (model_df[weight_col] > 0)
]

if len(model_df) < 5:
    st.error("At least 5 valid records are required to train this model.")
    st.stop()

# ---------------------------------
# 5. TRAIN LINEAR REGRESSION MODEL
# ---------------------------------
X = model_df[[anc_col]]
y = model_df[weight_col]

test_size = 0.2

if len(model_df) < 10:
    st.warning(
        "Your dataset contains fewer than 10 valid records. "
        "Evaluation results may be unreliable."
    )

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=test_size,
    random_state=42
)

model = LinearRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred) if len(y_test) >= 2 else np.nan

correlation = model_df[anc_col].corr(model_df[weight_col])

# ---------------------------------
# 6. HOME PAGE
# ---------------------------------
if page == "Home":

    st.header("Welcome")

    st.write("""
    This research application investigates the relationship between
    antenatal care visits and birth weight among babies delivered at
    the selected healthcare facility in Kenya.

    Linear Regression is used to estimate birth weight based on the
    number of ANC visits.
    """)

    col1, col2, col3 = st.columns(3)

    col1.metric("Total valid records", f"{len(model_df):,}")
    col2.metric("Average ANC visits", f"{model_df[anc_col].mean():.2f}")
    col3.metric("Average birth weight", f"{model_df[weight_col].mean():.1f} g")

    st.write("### Research variables")

    st.write("**Predictor:** Number of antenatal care visits (ANC Visits).")
    st.write("**Outcome:** Baby birth weight measured in grams.")
    st.write("**Machine Learning algorithm:** Linear Regression.")

# ---------------------------------
# 7. ABOUT THE RESEARCH
# ---------------------------------
elif page == "About the Research":

    st.header("About the Research")

    st.subheader("Research background")

    st.write("""
    Antenatal care is an important component of maternal healthcare.
    During ANC visits, pregnant women receive monitoring, health
    education, screening and interventions intended to support
    maternal and fetal health.

    This study investigates whether the number of ANC visits is
    associated with the birth weight of babies.
    """)

    st.subheader("General objective")

    st.write(
        "To investigate the effect of antenatal care visits on baby birth weight."
    )

    st.subheader("Specific objectives")

    st.write("""
    1. To examine the relationship between ANC visits and birth weight.
    2. To develop a Linear Regression model for predicting birth weight.
    3. To evaluate the performance of the model using MAE, RMSE and R².
    """)

    st.subheader("Research methodology")

    st.write("""
    The study uses maternity records. Data is cleaned and analysed
    before a Linear Regression model is trained using ANC visits as
    the predictor and birth weight as the target variable.
    """)

# ---------------------------------
# 8. DATASET PAGE
# ---------------------------------
elif page == "Dataset":

    st.header("Dataset Overview")

    st.write(f"**Uploaded file:** {selected_file.name}")

    col1, col2 = st.columns(2)

    col1.metric("Original dataset rows", len(df))
    col2.metric("Valid model records", len(model_df))

    st.subheader("Dataset preview")
    st.dataframe(df.head(20), use_container_width=True)

    st.subheader("Descriptive statistics")
    st.dataframe(model_df.describe(), use_container_width=True)

    st.subheader("Missing values in selected columns")
    st.dataframe(df[[anc_col, weight_col]].isnull().sum())

    st.write(
        f"Records excluded because of missing or invalid selected values: "
        f"{len(df) - len(model_df)}"
    )

# ---------------------------------
# 9. DATA ANALYSIS PAGE
# ---------------------------------
elif page == "Data Analysis":

    st.header("Exploratory Data Analysis")

    st.subheader("ANC Visits versus Birth Weight")

    fig, ax = plt.subplots(figsize=(9, 5))

    ax.scatter(
        model_df[anc_col],
        model_df[weight_col],
        alpha=0.6
    )

    ax.set_xlabel("Number of ANC Visits")
    ax.set_ylabel("Birth Weight (grams)")
    ax.set_title("Relationship between ANC Visits and Birth Weight")

    st.pyplot(fig)
    plt.close(fig)

    st.write(f"**Pearson correlation coefficient:** {correlation:.3f}")

    if pd.notna(correlation):
        if correlation > 0:
            st.write("The observed relationship is positive.")
        elif correlation < 0:
            st.write("The observed relationship is negative.")
        else:
            st.write("There is no linear correlation in this dataset.")

    st.subheader("Distribution of Birth Weight")

    fig, ax = plt.subplots(figsize=(9, 5))

    ax.hist(
        model_df[weight_col],
        bins=20,
        edgecolor="black"
    )

    ax.set_xlabel("Birth Weight (grams)")
    ax.set_ylabel("Number of Babies")
    ax.set_title("Distribution of Birth Weight")

    st.pyplot(fig)
    plt.close(fig)

    st.subheader("Low Birth Weight")

    lbw_count = (model_df[weight_col] < 2500).sum()
    normal_count = (model_df[weight_col] >= 2500).sum()

    col1, col2 = st.columns(2)

    col1.metric("Low birth weight (<2500 g)", int(lbw_count))
    col2.metric("Birth weight ≥2500 g", int(normal_count))

    st.caption(
        "Low birth weight is defined as birth weight below 2,500 grams."
    )

# ---------------------------------
# 10. PREDICTION PAGE
# ---------------------------------
elif page == "Birth Weight Prediction":

    st.header("Predict Baby Birth Weight")

    st.write(
        "Enter the number of ANC visits to obtain a birth weight estimate."
    )

    min_anc = float(model_df[anc_col].min())
    max_anc = float(model_df[anc_col].max())

    anc_value = st.number_input(
        "Number of ANC visits",
        min_value=0,
        max_value=100,
        value=max(0, int(model_df[anc_col].median())),
        step=1
    )

    if st.button("Predict Birth Weight"):

        prediction = model.predict(
            pd.DataFrame({anc_col: [anc_value]})
        )[0]

        st.subheader("Prediction result")

        st.metric(
            "Estimated birth weight",
            f"{prediction:,.0f} grams"
        )

        if prediction < 2500:
            st.warning(
                "The model estimate is below 2,500 grams, "
                "the low birth weight threshold."
            )
        else:
            st.success(
                "The model estimate is 2,500 grams or above."
            )

        if anc_value < min_anc or anc_value > max_anc:
            st.warning(
                "This ANC value is outside the range observed in the "
                "training dataset. Treat the estimate with caution."
            )

        st.caption(
            "This is a model-based estimate, not a clinical assessment."
        )

# ---------------------------------
# 11. MODEL EVALUATION PAGE
# ---------------------------------
elif page == "Model Evaluation":

    st.header("Linear Regression Model Evaluation")

    st.write(
        "The model is evaluated using a test dataset that was not "
        "used to train the model."
    )

    col1, col2, col3 = st.columns(3)

    col1.metric("MAE", f"{mae:.2f} g")
    col2.metric("RMSE", f"{rmse:.2f} g")
    col3.metric(
        "R² Score",
        f"{r2:.3f}" if pd.notna(r2) else "Not available"
    )

    st.subheader("Actual versus Predicted Birth Weight")

    results = pd.DataFrame({
        "Actual Birth Weight": y_test.values,
        "Predicted Birth Weight": y_pred
    })

    fig, ax = plt.subplots(figsize=(8, 6))

    ax.scatter(
        results["Actual Birth Weight"],
        results["Predicted Birth Weight"],
        alpha=0.7
    )

    min_value = min(
        results["Actual Birth Weight"].min(),
        results["Predicted Birth Weight"].min()
    )

    max_value = max(
        results["Actual Birth Weight"].max(),
        results["Predicted Birth Weight"].max()
    )

    ax.plot(
        [min_value, max_value],
        [min_value, max_value],
        linestyle="--"
    )

    ax.set_xlabel("Actual Birth Weight (grams)")
    ax.set_ylabel("Predicted Birth Weight (grams)")
    ax.set_title("Actual versus Predicted Birth Weight")

    st.pyplot(fig)
    plt.close(fig)

    st.subheader("Prediction results table")
    st.dataframe(results, use_container_width=True)

    st.subheader("Interpretation of evaluation metrics")

    st.write("""
    - **MAE:** Average absolute difference between actual and predicted
      birth weights. Lower values indicate smaller prediction errors.
    - **RMSE:** Measures prediction errors while giving greater weight
      to larger errors. Lower values are generally better.
    - **R²:** Indicates how much variation in birth weight is explained
      by the model. A value closer to 1 generally indicates better fit.
    """)

    st.warning(
        "These results evaluate a model using ANC visits as its only "
        "predictor. They do not establish a causal effect of ANC on birth weight."
    )

st.sidebar.divider()
st.sidebar.caption("ANC and Birth Weight Research Project")
