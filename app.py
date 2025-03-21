import joblib as jb
import streamlit as st

models = {
    "SVM Binary Classification": jb.load("svm_binary.pkl"),
    "SVM Multi Classification": jb.load("svm_multi.pkl"),
    "Logistic Binary Classification": jb.load("logistic_binary.pkl"),
    "Logistic ovr Classification": jb.load("logistic_ovr.pkl"),
    "Logistic Multinomial Classification": jb.load("logistic_multinomial.pkl")
}

st.title("Multi class model prediction")

selected_model = st.selectbox("Choose your model to predict",list(models.keys()))

sepal_length = st.number_input("Sepal Length (cm)", min_value=0.0, max_value=10.0, step=0.1, value=3.0)
sepal_width = st.number_input("Sepal Width (cm)", min_value=0.0, max_value=10.0, step=0.1, value=3.0)
petal_length = st.number_input("Petal Length (cm)", min_value=0.0, max_value=10.0, step=0.1, value=3.0)
petal_width = st.number_input("Petal Width (cm)", min_value=0.0, max_value=10.0, step=0.1, value=3.0)

prediction = None
if st.button("Predict"):
    model = models[selected_model]
    user_input = [[sepal_length, sepal_width, petal_length, petal_width]]
    prediction = model.predict(user_input)[0]
if prediction is not None:
    class_names = ["Setosa", "Versicolor", "Virginica"]
    st.success(f"The model predicts: {class_names[prediction]}")