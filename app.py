import streamlit as st
import pandas as pd
import joblib

# --------------------------
# CONFIG
# --------------------------
st.set_page_config(page_title="Susovan's SmartBank", layout="centered")

# --------------------------
# LOAD FILES
# --------------------------
model = joblib.load("models/model.pkl")
encoder = joblib.load("models/encoder.pkl")
features = joblib.load("models/features.pkl")
options = joblib.load("models/options.pkl")
state_city_map = joblib.load("models/state_city_map.pkl")

state_list = sorted(state_city_map.keys())

# --------------------------
# CSS (CLEAN UI)
# --------------------------
st.markdown("""
<style>
.stApp {
    background: linear-gradient(to right, #0f2027, #203a43, #2c5364);
    color: white;
}
.main > div {
    max-width: 900px;
    margin: auto;
}
.stButton>button {
    width: 100%;
    border-radius: 10px;
    height: 45px;
}
div[data-baseweb="select"] {
    color: black;
}
</style>
""", unsafe_allow_html=True)

# --------------------------
# SIDEBAR
# --------------------------
st.sidebar.title("🏦 Susovan's SmartBank")

page = st.sidebar.radio(
    "Navigation",
    ["📊 Dashboard", "🔍 Fraud Detection", "📜 History", "ℹ️ About"]
)

st.sidebar.markdown("---")
st.sidebar.caption("AI Fraud Detection System")

# --------------------------
# SEARCHABLE FUNCTION (SAFE KEYS)
# --------------------------
def searchable_select(label, options_list, key):
    search_key = f"{key}_search"
    select_key = f"{key}_select"

    search = st.text_input(f"🔍 Search {label}", key=search_key)

    if search:
        filtered = [x for x in options_list if search.lower() in str(x).lower()]
    else:
        filtered = options_list[:200]

    if not filtered:
        st.warning(f"No {label} found")
        return None

    return st.selectbox(label, filtered, key=select_key)

# --------------------------
# SESSION STATE
# --------------------------
if "history" not in st.session_state:
    st.session_state.history = []

# ==========================
# 📊 DASHBOARD
# ==========================
if page == "📊 Dashboard":

    st.title("📊 Fraud Monitoring Dashboard")

    total = len(st.session_state.history)
    frauds = sum(1 for x in st.session_state.history if x["Result"] == "Fraud")

    col1, col2 = st.columns(2)
    col1.metric("Total Transactions", total)
    col2.metric("Fraud Detected", frauds)

    if total > 0:
        df = pd.DataFrame(st.session_state.history)
        st.subheader("📈 Risk Trend")
        st.line_chart(df["Risk"])

# ==========================
# 🔍 FRAUD DETECTION
# ==========================
elif page == "🔍 Fraud Detection":

    st.title("🔍 Fraud Detection Engine")
    st.caption("Enter transaction details to evaluate risk")

    # Inputs
    amount = st.number_input("Amount", min_value=0.0, value=100.0)
    hour = st.slider("Hour", 0, 23, 12)
    age = st.number_input("Age", 18, 100, 30)

    gender = st.selectbox("Gender", ["M", "F"])
    category = st.selectbox("Category", options["category"])

    merchant = searchable_select("Merchant", options["merchant"], "merchant")

    state = st.selectbox("State", state_list)
    city = st.selectbox("City", state_city_map.get(state, []))

    job = searchable_select("Job", options["job"], "job")

    # Prediction
    if st.button("🚀 Predict"):

        if None in [merchant, city, state, job] or not category:
            st.warning("⚠️ Please fill all fields properly")
            st.stop()

        with st.spinner("Analyzing transaction..."):

            try:
                cat_vals = [[category, merchant, gender, city, state, job]]
                encoded = encoder.transform(cat_vals)[0]

                input_data = {
                    'amt': amount,
                    'hour': hour,
                    'age': age,
                    'category': encoded[0],
                    'merchant': encoded[1],
                    'gender': encoded[2],
                    'city': encoded[3],
                    'state': encoded[4],
                    'job': encoded[5]
                }

                # Ensure feature alignment
                for col in features:
                    if col not in input_data:
                        input_data[col] = 0

                df = pd.DataFrame([input_data])[features]

                prob = model.predict_proba(df)[0][1]

                # --------------------------
                # RESULT UI
                # --------------------------
                st.subheader("📊 Risk Analysis")

                st.progress(prob)
                st.metric("Fraud Risk", f"{prob:.2%}")

                # Risk level indicator
                if prob < 0.3:
                    st.success("🟢 Low Risk Transaction")
                elif prob < 0.7:
                    st.warning("🟡 Medium Risk Transaction")
                else:
                    st.error("🔴 High Risk Transaction")

                result = "Fraud" if prob > 0.5 else "Legit"

                if result == "Fraud":
                    st.error("🚨 Fraud Detected")
                else:
                    st.success("✅ Legitimate Transaction")

                # Save history
                st.session_state.history.append({
                    "Amount": amount,
                    "City": city,
                    "Category": category,
                    "Risk": round(prob, 2),
                    "Result": result
                })

            except Exception as e:
                st.error(f"Error: {e}")

# ==========================
# 📜 HISTORY
# ==========================
elif page == "📜 History":

    st.title("📜 Transaction History")

    if st.session_state.history:
        df = pd.DataFrame(st.session_state.history)
        st.dataframe(df, use_container_width=True)
    else:
        st.info("No transactions yet")

# ==========================
# ℹ️ ABOUT
# ==========================
else:

    st.title("ℹ️ About")

    st.write("""
    **Susovan's SmartBank Fraud Detection System**

    This application detects fraudulent credit card transactions using:
    
    - Random Forest Machine Learning Model  
    - Feature Engineering (including Age from DOB)  
    - Threshold Optimization  

    Built with:
    - Python
    - Scikit-learn
    - Streamlit
    """)

# --------------------------
# FOOTER
# --------------------------
st.markdown("---")
st.caption("© 2026 Susovan's SmartBank AI • Fraud Detection System")