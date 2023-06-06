from deepface import DeepFace
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import Model
def load_dataset(dataset_path):
    # Memuat dataset dan label Anda
    # Melakukan preprocessing yang diperlukan pada dataset

    # Membagi dataset menjadi set pelatihan dan pengujian
    X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, random_state=42)

    return X_train, X_test, y_train, y_test

def train_model(X_train, y_train):
    # Melatih model 
    # Melakukan augmentasi data atau preprocessing yang diperlukan pada X_train

    # Menginisialisasi model (misalnya, menggunakan TensorFlow, Keras, atau PyTorch)
    model = create_model()

    # Melatih model
    model.fit(X_train, y_train)

    return model

def evaluate_model(model, X_test, y_test):
    # Mengevaluasi model yang telah dilatih menggunakan X_test dan y_test
    y_pred = model.predict(X_test)

    # Menampilkan laporan klasifikasi
    report = classification_report(y_test, y_pred)
    print(report)


if __name__ == "__main__":
    dataset_path = 'your_dataset_path'
    X_train, X_test, y_train, y_test = load_dataset(dataset_path)
    model = train_model(X_train, y_train)
    evaluate_model(model, X_test, y_test)
    model.save('emotion_model.h5')
