from collections import Counter


class MajorityClassifier:
    def fit(self, labels):
        self.majority_class = Counter(labels).most_common(1)[0][0]

    def predict(self, messages):
        return [self.majority_class for _ in messages]
