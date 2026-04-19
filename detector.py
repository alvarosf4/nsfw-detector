import tensorflow_hub as hub
import cv2
import numpy as np
from tensorflow.keras.models import load_model

# 🔹 Cargar modelo (asegúrate que el archivo esté en la misma carpeta)
model = load_model("modelo.h5", custom_objects={'KerasLayer': hub.KerasLayer})
# 🔹 Clases del modelo
classes = ['drawings', 'hentai', 'neutral', 'porn', 'sexy']

def analizar_imagen(ruta):
    # Cargar imagen
    img = cv2.imread(ruta)

    if img is None:
        print("❌ No se pudo cargar la imagen. Verifica la ruta.")
        return

    # Preprocesamiento
    img = cv2.resize(img, (224, 224))
    img = img.astype("float32") / 255.0
    img = np.reshape(img, (1, 224, 224, 3))

    # Predicción
    pred = model.predict(img)[0]

    print("\n📊 Resultados:")
    for i in range(len(classes)):
        print(f"{classes[i]}: {pred[i]*100:.2f}%")

    # 🔥 Lógica mejorada (más realista)
    nsfw_score = pred[1] + pred[3] + pred[4]  # hentai + porn + sexy

    print(f"\n🧠 Score NSFW: {nsfw_score*100:.2f}%")

    if nsfw_score > 0.6:
        print("🔴 Resultado: NSFW")
    else:
        print("🟢 Resultado: Seguro")


# 🔹 Ejecutar prueba
if __name__ == "__main__":
    ruta_imagen = input("📁 Ingresa la ruta de la imagen: ")
    analizar_imagen(ruta_imagen)