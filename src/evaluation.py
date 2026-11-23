
import random
from .instances import TrainingSet
from .preprocessing import Preprocessor


class Evaluator:
    """
    Class used to perform classifier evaluation
    """
    @staticmethod
    def mean(lst):
        return sum(lst) / len(lst) # Gets the mean of a given list

    @staticmethod
    def std(lst, mean):
        var = 0
        for xn in lst:
            var += (xn - mean)**2

        var = var / (len(lst) - 1)

        return var**0.5 # Gets the standard deviation from a given list
    
    def get_stats(self, lst):
        return self.mean(lst), self.std(lst, self.mean(lst))

    @staticmethod
    def training_accuracy(clf, training_set):
        """
        Gets the percentage accuracy over a training set (labeled instances)
        """
        score = 0

        for instance in training_set:
            correct_label = instance.label
            prediction = clf.predict(instance)

            if prediction == correct_label:
                score += 1

        score = score / len(training_set)
        return score

    @staticmethod
    def get_wrong(clf, training_set):
        """
        Gets the number of wrong classifications in each class
        Note: function is unused here, was previously used in some extraneuous model evaluation
        """
        wrong = {"A": 0, "B": 0, "E": 0, "V": 0}
        for instance in training_set:
            prediction = clf.predict(instance)
            if instance.label != prediction:
                wrong[instance.label] += 1

        return wrong

    @staticmethod
    def split(training_set, n_splits=10):
        """
        Helper method for cross validation, splits training set into n_split stratified folds
        """
        by_class = training_set.separate_by_class()
        folds = [TrainingSet() for i in range(n_splits)]
        classes = ["A", "V", "B", "E"]
        max_length = int(len(training_set) / n_splits) # maximum length of fold

        for label in classes:
            # Adds instances to folds equally by class
            class_set = by_class[label]

            i = 0
            while i < len(class_set):
                idx = i % n_splits
                instance = class_set[i]

                for j in range(n_splits):
                    jdx = (idx + j) % n_splits
                    if len(folds[jdx]) < max_length:
                        # In the case that fold is already of maximum length, adds instance to the next fold
                        folds[jdx].add_instance(instance)
                        break

                    if j == n_splits:
                        # All folds are of maximum length, but there is an instance left
                        # Occurs when number of folds does not divide length of training set equally.
                        print("Something has gone terribly wrong")

                i += 1

        return folds

    def cross_validation(self, clf, input_set, n_words=1000, remove_useless=False, random_state=11052021):
        """
        Evaluates a model using 10-fold cv. Note that preprocessing is part of the training
        process.
        """
        print("Cross Validating {}...".format(clf))
        random.seed(random_state)
        training_set = input_set.copy()
        random.shuffle(training_set.instances) # shuffle instances for random split
        folds = self.split(training_set)

        scores = []
        for i in range(10):
            test_set = folds[i]
            training_folds = [fold for fold in folds if fold != test_set] # Get 9 training folds
            training_set = TrainingSet()
            for fold in training_folds:
                training_set += fold
                
            # Preprocess
            preprocessor = Preprocessor()
            best_words = preprocessor.set_most_frequent_words(training_set, n_words, remove_useless=remove_useless)
            processed_training_set = preprocessor.process_training_set(training_set)

            processed_test_set = preprocessor.process_training_set(test_set) # Only allow most common words

            clf.train(processed_training_set, best_words)

            score = self.training_accuracy(clf, processed_test_set) # test classifier in fold left out
            scores.append(score)

            print("  -> Completed {}...".format(i+1))

        return scores




