import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.metrics import classification_report, confusion_matrix
import joblib

# Prophet for forecasting
try:
    from prophet import Prophet
    PROPHET_AVAILABLE = True
except:
    try:
        from fbprophet import Prophet
        PROPHET_AVAILABLE = True
    except:
        PROPHET_AVAILABLE = False

# ---------------------------
# Page setup
# ---------------------------
st.set_page_config(page_title="Crime Dashboard", layout="wide")

# ---------------------------
# Glassmorphism CSS
# ---------------------------
st.markdown(
    """
    <style>
    body {
        background: linear-gradient(135deg, #1f1c2c, #928dab);
        color: white;
    }
    .stApp {
        background: transparent;
    }
    .glass-card {
        background: rgba(255, 255, 255, 0.1);
        border-radius: 20px;
        padding: 20px;
        backdrop-filter: blur(15px);
        -webkit-backdrop-filter: blur(15px);
        box-shadow: 0 8px 32px 0 rgba(31, 38, 135, 0.37);
        border: 1px solid rgba(255, 255, 255, 0.18);
        margin-bottom: 20px;
    }
    h1, h2, h3, h4 {
        color: #f2f2f2 !important;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# ---------------------------
# Title
# ---------------------------
st.markdown('<div class="glass-card"><h1>📊 Crime Analytics Dashboard</h1></div>', unsafe_allow_html=True)

# ---------------------------
# File Upload
# ---------------------------
crime_file = st.file_uploader("Upload Crime Dataset CSV", type="csv")

if crime_file:
    df = pd.read_csv(crime_file)

    # ---------------------------
    # Filter Controls
    # ---------------------------
    st.sidebar.header("🔍 Filters")

    crime_types = df["Crime"].unique() if "Crime" in df.columns else []
    selected_crime = st.sidebar.selectbox("Crime Category", options=["All"] + list(crime_types))

    provinces = df["Province"].unique() if "Province" in df.columns else []
    selected_province = st.sidebar.selectbox("Province", options=["All"] + list(provinces))

    years = sorted(df["Year"].unique()) if "Year" in df.columns else []
    year_range = st.sidebar.slider(
        "Select Year Range",
        min_value=int(min(years)),
        max_value=int(max(years)),
        value=(int(min(years)), int(max(years)))
    )

    # Apply filters
    filtered_df = df.copy()
    if selected_crime != "All":
        filtered_df = filtered_df[filtered_df["Crime"] == selected_crime]
    if selected_province != "All":
        filtered_df = filtered_df[filtered_df["Province"] == selected_province]
    filtered_df = filtered_df[(filtered_df["Year"] >= year_range[0]) & (filtered_df["Year"] <= year_range[1])]

    # ---------------------------
    # EDA Section
    # ---------------------------
    st.markdown('<div class="glass-card"><h2>📈 Exploratory Data Analysis (EDA)</h2>', unsafe_allow_html=True)

    if "Year" in filtered_df.columns and "Incidents" in filtered_df.columns:
        yearly = filtered_df.groupby("Year")["Incidents"].sum()
        st.line_chart(yearly)

    if "Province" in filtered_df.columns and "Incidents" in filtered_df.columns:
        prov = filtered_df.groupby("Province")["Incidents"].sum().sort_values(ascending=False)
        st.bar_chart(prov)

    st.dataframe(filtered_df.head(20))
    st.markdown('</div>', unsafe_allow_html=True)

    # ---------------------------
    # Classification Section
    # ---------------------------
    st.markdown('<div class="glass-card"><h2>🤖 Hotspot Classification</h2>', unsafe_allow_html=True)

    if "Incidents" in filtered_df.columns:
        df_clf = filtered_df.select_dtypes(include=[np.number]).copy()
        if df_clf.shape[1] > 1:
            df_clf["hotspot"] = (df_clf["Incidents"] >= df_clf["Incidents"].quantile(0.75)).astype(int)

            X = df_clf.drop(columns=["hotspot"])
            y = df_clf["hotspot"]

            imp = SimpleImputer(strategy="median")
            X_imp = pd.DataFrame(imp.fit_transform(X), columns=X.columns)

            X_train, X_test, y_train, y_test = train_test_split(X_imp, y, test_size=0.2, stratify=y, random_state=42)

            clf = RandomForestClassifier(n_estimators=200, class_weight="balanced", random_state=42)
            clf.fit(X_train, y_train)
            y_pred = clf.predict(X_test)

            st.subheader("Performance Metrics")
            st.text(classification_report(y_test, y_pred))

            cm = confusion_matrix(y_test, y_pred)
            st.subheader("Confusion Matrix")
            st.write(cm)

            joblib.dump(clf, "rf_hotspot_model.joblib")
            st.success("✅ Model trained & saved as rf_hotspot_model.joblib")

    st.markdown('</div>', unsafe_allow_html=True)

    # ---------------------------
    # Forecasting Section
    # ---------------------------
    st.markdown('<div class="glass-card"><h2>⏳ Time Series Forecasting</h2>', unsafe_allow_html=True)

    if not PROPHET_AVAILABLE:
        st.error("⚠️ Prophet not installed. Please run: pip install prophet")
    else:
        if "Year" in filtered_df.columns and "Incidents" in filtered_df.columns:
            ts = filtered_df.groupby("Year")["Incidents"].sum().reset_index()
            ts["ds"] = pd.to_datetime(ts["Year"].astype(str) + "-01-01")
            ts = ts.rename(columns={"Incidents": "y"})[["ds", "y"]]

            if ts.shape[0] >= 5:
                m = Prophet(yearly_seasonality=True)
                m.fit(ts)
                future = m.make_future_dataframe(periods=5, freq="Y")
                fcst = m.predict(future)

                fig1 = m.plot(fcst)
                st.pyplot(fig1)

                st.write("Forecast results with confidence intervals:")
                st.dataframe(fcst[["ds", "yhat", "yhat_lower", "yhat_upper"]].tail(10))
            else:
                st.warning("⚠️ Not enough data points to forecast.")

    st.markdown('</div>', unsafe_allow_html=True)

else:
    st.info("📂 Please upload your crime dataset to begin.")
