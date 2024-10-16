import unittest
import numpy as np
from spam import calculate_priors, calculate_likelihoods, predict, evaluate
from sklearn.feature_extraction.text import CountVectorizer

class TestSpamClassifier(unittest.TestCase):
    
    def setUp(self):
        
        messages = []
        labels = []
        
        with open("messages.txt", "r") as file:
            for line in file:
                parts = line.strip().split(',')
                if len(parts) == 2:
                    message, label = parts
                    messages.append(message)
                    labels.append(label)
        
        self.messages = np.array(messages)
        self.labels = np.array(labels)
        
        # Initialize CountVectorizer and transform messages
        self.cv = CountVectorizer()
        self.X = self.cv.fit_transform(self.messages)
        self.vocab = self.cv.get_feature_names_out()
        
        # Manually create training and testing sets
        np.random.seed(42)  # For reproducibility
        indices = np.random.permutation(len(self.messages))
        split_index = int(0.67 * len(self.messages))
        train_indices = indices[:split_index]
        test_indices = indices[split_index:]
        
        self.X_train = self.X[train_indices].toarray()
        self.X_test = self.X[test_indices].toarray()
        self.y_train = self.labels[train_indices]
        self.y_test = self.labels[test_indices]
        
    def test_calculate_priors(self):
        priors = calculate_priors(self.y_train)
        self.assertIn("spam", priors)
        self.assertIn("not_spam", priors)
        self.assertAlmostEqual(priors["spam"] + priors["not_spam"], 1.0, places=2)
    
    def test_calculate_likelihoods(self):
        likelihoods = calculate_likelihoods(self.X_train, self.y_train, self.vocab)
        self.assertIn("spam", likelihoods)
        self.assertIn("not_spam", likelihoods)
        self.assertEqual(len(likelihoods["spam"]), len(self.vocab))  # Ensure vocab size matches likelihood size
    
    def test_predict(self):
        priors = calculate_priors(self.y_train)
        likelihoods = calculate_likelihoods(self.X_train, self.y_train, self.vocab)
        message = "Win a free prize!"
        prediction = predict(message, priors, likelihoods, self.vocab)
        self.assertEqual(prediction, "spam")
    
    def test_evaluate(self):
        priors = calculate_priors(self.y_train)
        likelihoods = calculate_likelihoods(self.X_train, self.y_train, self.vocab)
        accuracy = evaluate(self.X_test, self.y_test, priors, likelihoods, self.vocab)
        self.assertGreaterEqual(accuracy, 0.5)  # Expect accuracy to be at least 50%

if __name__ == "__main__":
    unittest.main()
