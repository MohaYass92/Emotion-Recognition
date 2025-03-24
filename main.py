from scripts.data_preprocessing import load_data
from scripts.features_extraction import extract_features_from_generator
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

def main():
    # Load data
    train_generator, test_generator = load_data()

    # Extract features and labels
    train_features, train_labels = extract_features_from_generator(train_generator)
    test_features, test_labels = extract_features_from_generator(test_generator)

    # Train SVM model
    model = SVC(kernel='linear')
    model.fit(train_features, train_labels)

    # Predict
    test_predictions = model.predict(test_features)

    # Evaluate
    accuracy = accuracy_score(test_labels, test_predictions)
    print(f"Accuracy on test data: {accuracy * 100:.2f}%")

if __name__ == "__main__":
    main()
