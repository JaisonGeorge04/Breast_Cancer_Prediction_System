import joblib
from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import GradientBoostingClassifier

def train_model():
    print("Loading breast cancer dataset...")
    data = load_breast_cancer()
    X = data.data
    y = data.target

    print("Loading scaler...")
    scaler = joblib.load('scaler.pkl')

    print("Scaling dataset features...")
    X_scaled = scaler.transform(X)

    print("Training GradientBoostingClassifier...")
    # Using the same architecture parameters as the original model
    model = GradientBoostingClassifier(
        n_estimators=100,
        learning_rate=0.1,
        max_depth=3,
        random_state=42
    )
    model.fit(X_scaled, y)

    print("Saving model to gradient_boosting_model.pkl...")
    joblib.dump(model, 'gradient_boosting_model.pkl')
    print("Model trained and saved successfully!")

if __name__ == '__main__':
    train_model()
