from tensorflow.keras.models import load_model
import pickle


# Загружаем модель
model = load_model("C:\\python_projects\\sales_forecast\\model\\model.h5")
model.save("C:\\python_projects\\sales_forecast\\model\\model.keras")

