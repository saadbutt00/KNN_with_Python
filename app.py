import streamlit as st
import numpy as np
import collections

st.set_page_config(page_title="k-NN", layout="centered")

st.title("K-Nearest Neighbors (KNN) Classifier Model")

# ---- Description ----
st.markdown("""
KNN works by classifying a new data point based on the "k" nearest data points in the dataset. 
KNN operates on the principle that similar data points are close to each other in a feature space. 
**Example**: KNN can be used in agriculture to classify plants based on their features, such as 
             leaf shape, size, and color, and predict their growth rate and yield. 
""")

st.markdown("""
**Step 1**: Enter X's values - You can also name X column with your desired one.

**Step 2**: Enter Y values - It's length should be equal to the length of X feature's values.

**Step 3**: Select your desired K value.

**Step 4**: Click 'Submit' & your model will be trained.

**Step 5**: It will appear predict value side where you will enter your sample values & get your predicted answer.
""")

# ---- Input Section ----
num_feature = st.number_input("Enter Number of Features", min_value=1, step=1)
k = st.number_input("Select K", min_value=1, step=1)

feature_names, feature_values, ready = [], [], False

if num_feature:
    st.subheader("🔧 Feature Names")
    for i in range(num_feature):
        name = st.text_input(f"Name of Feature {i+1}", value=f"X{i}")
        feature_names.append(name)

    target_name = st.text_input("Enter name for Target Feature (Y)", value="Target")

    st.subheader("📥 Enter Training Data")
    all_entered = True
    for name in feature_names:
        values = st.text_input(f"Values for `{name}` (space-separated)").split()
        if values:
            feature_values.append(values)
        else:
            all_entered = False

    y_vals = st.text_input(f"Values for `{target_name}` (space-separated)").split()
    if y_vals and all_entered:
        if all(len(fv) == len(y_vals) for fv in feature_values):
            ready = True
        else:
            st.warning("Mismatch: Number of values in features and target must match.")

# ---- Submit and Store Model ----
if ready and st.button("Submit"):
    updated_X = []
    is_numeric = []
    for col in feature_values:
        numeric_col = []
        try:
            numeric_col = list(map(float, col))
            updated_X.append(numeric_col)
            is_numeric.append(True)
        except:
            updated_X.append(col)
            is_numeric.append(False)

    numeric_X = [updated_X[i] for i in range(len(updated_X)) if is_numeric[i]]
    mixed_X = np.transpose(updated_X).tolist()
    numeric_X = np.transpose(numeric_X).tolist()

    st.session_state["model"] = {
        "X_all": mixed_X,
        "X_numeric": numeric_X,
        "Y": [y.lower() for y in y_vals],
        "k": k,
        "feature_names": feature_names,
        "target_name": target_name,
        "is_numeric": is_numeric
    }
    st.success("✅ Model Trained! Ready to predict.")

# ---- Prediction ----
if "model" in st.session_state:
    st.subheader("📐 Select Distance Metric")
    dist_opt = st.selectbox("Distance Metric", ["Euclidean", "Manhattan", "Minkowski"])
    if dist_opt == "Minkowski":
        p_val = st.number_input("Enter p value", min_value=1.0, step=0.5)
    else:
        p_val = None

    st.subheader("🔍 Enter Sample to Predict")
    user_sample = []
    for i, name in enumerate(st.session_state["model"]["feature_names"]):
        val = st.text_input(f"Enter value for `{name}`", key=f"sample_{name}")
        user_sample.append(val)

    if st.button("Predict"):
        # Filter numeric input
        numeric_input = []
        for val, is_num in zip(user_sample, st.session_state["model"]["is_numeric"]):
            if is_num:
                try:
                    numeric_input.append(float(val))
                except:
                    st.error(f"`{val}` is not a valid number.")
                    st.stop()

        if len(numeric_input) != sum(st.session_state["model"]["is_numeric"]):
            st.error("Error: Sample size must match number of numeric features.")
            st.stop()

        # Distance functions
        def euclidean_dist(x1, x2):
            return sum((a - b) ** 2 for a, b in zip(x1, x2)) ** 0.5

        def manhattan_dist(x1, x2):
            return sum(abs(a - b) for a, b in zip(x1, x2))

        def minkowski_dist(x1, x2, p):
            return (sum(abs(a - b) ** p for a, b in zip(x1, x2))) ** (1 / p)

        def get_distance(x1, x2):
            if dist_opt == "Euclidean":
                return euclidean_dist(x1, x2)
            elif dist_opt == "Manhattan":
                return manhattan_dist(x1, x2)
            elif dist_opt == "Minkowski":
                return minkowski_dist(x1, x2, p_val)

        Xn = st.session_state["model"]["X_numeric"]
        Yn = st.session_state["model"]["Y"]
        k = st.session_state["model"]["k"]

        # Edge case protection for k
        k_use = min(int(k), len(Xn))
        if k_use == 0:
            st.error("Not enough data points for prediction.")
            st.stop()

        # Predict
        distances = [(get_distance(numeric_input, xi), y) for xi, y in zip(Xn, Yn)]
        neighbors = sorted(distances, key=lambda x: x[0])[:k_use]
        labels = [y for _, y in neighbors]

        if labels:
            prediction = collections.Counter(labels).most_common(1)[0][0]
            st.success(f"🔮 Prediction for `{st.session_state['model']['target_name']}`: `{prediction}`")
        else:
            st.warning("⚠ No neighbors found. Cannot predict.")

        # Show non-numeric (string) features
        str_feats = [name for name, is_num in zip(st.session_state["model"]["feature_names"], st.session_state["model"]["is_numeric"]) if not is_num]
        if str_feats:
            st.markdown("### String Features (Not used in distance, but tracked)")
            for i, name in enumerate(str_feats):
                st.markdown(f"- `{name}`")