import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier 
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score
from sklearn.preprocessing import LabelEncoder

class ModelPipeline:
    def __init__(self):
        self.rf_model = None  
        self.lr_model = None
        self.selected_model = None
        self.data = None
        self.test_data = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.label_encoders = {}

    def load(self, file_path):
        """Loads data from an Excel file."""
        self.data = pd.read_excel(file_path)
        print("Data loaded successfully.")
        #here we had loaded the dataset for training
    def load_test_data(self, file_path):
        """Loads external test data from an Excel file."""
        self.test_data = pd.read_excel(file_path)
        print("Test data loaded successfully.")
        #here we had loaded the test dataset

    def preprocess(self):
        """Preprocesses the data for training."""
        if self.data is None:
            raise ValueError("Data not loaded. Please load data first.")
        #preprocessing of dataset is done here of training data
        # Handle datetime columns by converting them to numerical format
        for col in self.data.select_dtypes(include=['datetime64', 'datetime']):
            self.data[col] = self.data[col].apply(lambda x: x.timestamp() if pd.notnull(x) else 0)
        
        # Handle categorical columns by encoding them
        for col in self.data.select_dtypes(include=['object', 'category']):
            le = LabelEncoder()
            self.data[col] = le.fit_transform(self.data[col].astype(str))
            self.label_encoders[col] = le
        
        # Assuming the last column is the target variable
        X = self.data.iloc[:, :-1]
        y = self.data.iloc[:, -1]
        
        # Train-test split
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        print("Data preprocessing complete.")

    def preprocess_test_data(self):
        """Preprocesses the external test data."""
        #preprocessing of dataset is done here of testing data
        if self.test_data is None:
            raise ValueError("External test data not loaded. Please load test data first.")
        
        # Handle datetime columns by converting them to numerical format
        for col in self.test_data.select_dtypes(include=['datetime64', 'datetime']):
            self.test_data[col] = self.test_data[col].apply(lambda x: x.timestamp() if pd.notnull(x) else 0)
        
        # Handle categorical columns by encoding them using the same label encoders as training data
        for col in self.test_data.select_dtypes(include=['object', 'category']):
            if col in self.label_encoders:
                le = self.label_encoders[col]
                self.test_data[col] = le.transform(self.test_data[col].astype(str).fillna("unknown"))
            else:
                raise ValueError(f"Column '{col}' in test data was not present in training data.")
        print("External test data preprocessing complete.")

    def train(self):
        """Trains both RandomForestClassifier and LogisticRegression models."""
        if self.X_train is None or self.y_train is None:
            raise ValueError("Data not preprocessed. Please preprocess data first.")
        
        # Train RandomForestClassifier (changed from AdaBoostClassifier)
        self.rf_model = RandomForestClassifier(random_state=42)
        self.rf_model.fit(self.X_train, self.y_train)
        print("Random Forest training complete.")
        
        # Train LogisticRegression
        self.lr_model = LogisticRegression(random_state=42, max_iter=1000)
        self.lr_model.fit(self.X_train, self.y_train)
        print("Logistic Regression training complete.")

    def test(self):
        """Tests both models and selects the one with higher accuracy."""
        if self.rf_model is None or self.lr_model is None:
            raise ValueError("Models not trained. Please train the models first.")
        
        # Test RandomForestClassifier (changed from AdaBoostClassifier)
        rf_predictions = self.rf_model.predict(self.X_test)
        rf_accuracy = accuracy_score(self.y_test, rf_predictions)
        print("Random Forest Testing Results:")
        print(f"Accuracy: {rf_accuracy:.2f}")
        
        # Test LogisticRegression
        lr_predictions = self.lr_model.predict(self.X_test)
        lr_accuracy = accuracy_score(self.y_test, lr_predictions)
        print("Logistic Regression Testing Results:")
        print(f"Accuracy: {lr_accuracy:.2f}")
        
        # here we are Selecting the model with higher accuracy
        if rf_accuracy >= lr_accuracy:
            self.selected_model = self.rf_model
            print("Random Forest selected as the final model.")
        else:
            self.selected_model = self.lr_model
            print("Logistic Regression selected as the final model.")

    def test_external(self):
        """Tests the selected model on external test data if provided."""
        if self.selected_model is None:
            raise ValueError("No model selected. Please run the test method first.")
        
        self.preprocess_test_data()
        
        # Assuming the last column is the target variable in external test data
        X_external = self.test_data.iloc[:, :-1]
        y_external = self.test_data.iloc[:, -1]
        
        # Testing the selected model on external data
        predictions = self.selected_model.predict(X_external)
        accuracy = accuracy_score(y_external, predictions)
        report = classification_report(y_external, predictions)
        print("Selected Model External Testing Results:")
        print(f"Accuracy: {accuracy:.2f}")
        print("Classification Report:")
        print(report)

    def predict(self, input_data):
        """Generates predictions for the provided input data using the selected model."""
        if self.selected_model is None:
            raise ValueError("No model selected. Please run the test method first.")
        
        predictions = self.selected_model.predict(input_data)
        return predictions

if __name__ == "__main__":
    pipeline = ModelPipeline()
    
    # File path for the training data
    train_file_path = "train_data.xlsx"
    
    # File path for the external test data
    test_file_path = "test_data.xlsx"
    
    # Execute the pipeline
    pipeline.load(train_file_path)
    pipeline.preprocess()
    pipeline.train()
    pipeline.test()
    
    # Load and test on external data
    pipeline.load_test_data(test_file_path)
    pipeline.test_external()
    
    #example
    example_data = pipeline.X_test.iloc[:5]  # Example input data
    predictions = pipeline.predict(example_data)
    print("Predictions:", predictions)

'''
OUTPUT:
Data loaded successfully.
Data preprocessing complete.
Random Forest training complete.
Logistic Regression training complete.
Random Forest Testing Results:
Accuracy: 0.76
Logistic Regression Testing Results:
Accuracy: 0.74
Random Forest selected as the final model.
Test data loaded successfully.
External test data preprocessing complete.
Selected Model External Testing Results:
Accuracy: 0.66
Classification Report:
              precision    recall  f1-score   support

           0       0.56      0.35      0.43      3055
           1       0.70      0.84      0.76      5400

    accuracy                           0.66      8455
   macro avg       0.63      0.59      0.59      8455
weighted avg       0.65      0.66      0.64      8455

Predictions: [1 1 1 1 1]
'''