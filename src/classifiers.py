import math as m
from .instances import TrainingSet


class NaiveBayesClassifier:
    """
    Standard Naive Bayes Classifier
    """
    def __init__(self):
        """
        Likelihoods stored as a dictionary of dictionary. Specifically:
        self.likelihoods[word][label] := estimated P(word|label)
        
        Priors stored as dictionary. Specifically:
        self.priors[label] := estimated P(label)
        """
        self.likelihoods = {}
        self.priors = {}

    def __repr__(self):
        return "Naive Bayes"

    def count(self, word, instance_set):
        """
        Gets the counts for a particular word in an instance set
        Used for estimating the conditional likelihoods
        """
        count = 0
        for instance in instance_set:
            instance_counts = instance.get_word_freq()
            if word in instance_counts:
                count += self.transform(instance_counts[word], word) # Transformation may be changed in extensions

        return count

    def transform(self, freq, word):
        """
        For frequency transformation extensions
        """
        return freq

    def set_likelihoods(self, training_set, words_to_use):
        """
        Estimate the conditional likelihoods from the training set
        """
        instances_by_class = training_set.separate_by_class()
        classes = ["A", "B", "E", "V"]
        self.likelihoods = {word: {} for word in words_to_use}

        for label in classes:
            word_counts = {} # Words counts across all instance in this class
            for word in words_to_use:
                word_counts[word] = self.count(word, instances_by_class[label]) + 1

            total = sum(word_counts.values())

            for word in self.likelihoods:
                self.likelihoods[word][label] = word_counts[word] / total

        return self.likelihoods

    def set_prior(self, training_set):
        """
        Estimates class priors for the training set
        """
        instances_by_class = training_set.separate_by_class()
        classes = ["A", "B", "E", "V"]
        total = len(training_set)

        for label in classes:
            value = len(instances_by_class[label])/total
            self.priors[label] = value

        return self.priors

    def train(self, training_set, words_to_use):
        """
        Trains model by setting likelihoods and priors
        """
        self.set_likelihoods(training_set, words_to_use)
        self.set_prior(training_set)

    def predict(self, instance):
        """
        Given an instance predicts the class based on the learned parameters
        """
        classes = ["A", "B", "E", "V"]
        log_probs = {label:0 for label in classes}
        word_counts = instance.get_word_freq()

        for label in classes:
            # Use logarithms to avoid underflow
            log_prior = m.log(self.priors[label])

            log_likelihood = 0
            for word in word_counts:
                if word not in self.likelihoods:
                    continue

                freq = word_counts[word]
                freq = self.transform(freq, word) # frequency transform for extensions
                prob = self.likelihoods[word][label]
                log_likelihood += freq * m.log(prob)

            log_posterior = log_prior + log_likelihood
            log_probs[label] = log_posterior
            
        # Gets maximum label for class over stored log(P(class|words)) (posteriors)
        max_label = "A"
        max_prob = log_probs["A"]
        for label in classes[1:]:
            if log_probs[label] > max_prob:
                max_prob = log_probs[label]
                max_label = label

        return max_label

    def get_predictions(self, test_set):
        """
        Given a test set returns a list of predictions
        """
        predictions = []

        for instance in test_set:
            predicted_label = self.predict(instance)
            prediction = (instance.id, predicted_label)
            predictions.append(prediction)

        return predictions


class TermTransformedNaiveBayesClassifier(NaiveBayesClassifier):
    """
    Logarithmic Term Frequency Transformation:
    freq = log(1+freq)
    """
    def __repr__(self):
        return "Term frequency transformed Naive Bayes"

    def transform(self, freq, word):
        """
        Frequency Transformation
        """
        return m.log(1 + freq)


class DocumentTransformedNaiveBayesClassifier(NaiveBayesClassifier):
    """
    Inverse Document Frequency Transformation
    """
    def __init__(self):
        """
        freq_weights holds the frequency of words across all documents (abstracts), only 
        accounting for their presence - not counting how many times they occur in total.
        
        The logs of these are stored for ease of computation.
        """
        NaiveBayesClassifier.__init__(self)
        self.freq_weights = {}

    def __repr__(self):
        return "Document frequency transformed Naive Bayes"
    def transform(self, freq, word):
        return freq * self.freq_weights[word]

    def set_frequency_weights(self, training_set, words_to_use):
        """
        Part of training process to learn word document frequency to use in the frequency transformation
        """
        self.freq_weights = {word: None for word in words_to_use}
        instance_number = len(training_set)
        for word in words_to_use:

            word_freq = 0
            for instance in training_set:
                counts = instance.get_word_freq()
                if word in counts:
                    word_freq += 1

            self.freq_weights[word] = m.log(instance_number/word_freq) # stores in log form for ease of computation

        return

    def train(self, training_set, words_to_use):
        self.set_frequency_weights(training_set, words_to_use) # Trains frequency weights first
        NaiveBayesClassifier.train(self, training_set, words_to_use)
        return    


class TransformedNaiveBayesClassifier(DocumentTransformedNaiveBayesClassifier):
    """
    Combines logarithmic term frequency and document frequency transformations
    Child class of DocumentTransformedNaiveBayesClassifier so only logarithmic transformations
    need implementation.
    
    Also includes hyperparameter d for optimizing
    """
    def __init__(self, d=1):
        DocumentTransformedNaiveBayesClassifier.__init__(self)
        self.d = d

    def __repr__(self):
        return "TF-IDF transformed Naive Bayes"

    def transform(self, freq, word):
        freq = m.log(self.d + freq) * self.freq_weights[word]
        return freq


# ==============================================================================================
#                        CLASSIFIERS BASED ON COMPLEMENT NAIVE BAYES
# ==============================================================================================

class ComplementNaiveBayesClassifier(NaiveBayesClassifier):
    def __repr__(self):
        return "Complement Naive Bayes"

    def set_likelihoods(self, training_set, words_to_use):
        """
        Likelihoods represent P(word|not class)
        """
        instances_by_class = training_set.separate_by_class()
        classes = ["A", "B", "E", "V"]
        self.likelihoods = {word: {} for word in words_to_use}

        for label in classes:
            word_counts = {}
            instances_not_class = TrainingSet()
            for comp_label in classes:
                # Construct training set consisting of all 'not label' instances
                if comp_label != label:
                    instances_not_class += instances_by_class[comp_label]

            for word in words_to_use:
                word_counts[word] = self.count(word, instances_not_class) + 1

            total = sum(word_counts.values())

            for word in self.likelihoods:
                self.likelihoods[word][label] = word_counts[word] / total

        return self.likelihoods

    def predict(self, instance):
        classes = ["A", "B", "E", "V"]
        log_probs = {label: 0 for label in classes}
        word_counts = instance.get_word_freq()

        for label in classes:
            log_prior = m.log(self.priors[label])

            log_likelihood = 0
            for word in word_counts:
                freq = word_counts[word]
                freq = self.transform(freq, word)
                prob = self.likelihoods[word][label]
                log_likelihood += freq * m.log(prob)

            log_posterior = log_prior - log_likelihood  # Notice subtract likelihood
            log_probs[label] = log_posterior

        max_label = "A"
        max_prob = log_probs["A"]
        for label in classes[1:]:
            if log_probs[label] > max_prob:
                max_prob = log_probs[label]
                max_label = label

        return max_label


class DocumentTransformedComplementNaiveBayesClassifier(ComplementNaiveBayesClassifier):
    """
    Complement Naive Bayes Classifier with Document Frequency Transformation
    """
    def __init__(self):
        NaiveBayesClassifier.__init__(self)
        self.freq_weights = {}

    def __repr__(self):
        return "Document frequency transformed Complement Naive Bayes"

    def transform(self, freq, word):
        """
        Document frequency transformation
        """
        freq = freq * self.freq_weights[word]
        return freq

    def set_frequency_weights(self, training_set, words_to_use):

        self.freq_weights = {word: None for word in words_to_use}
        instance_number = len(training_set)
        for word in words_to_use:

            word_freq = 0  # 0 for best classifier so far
            for instance in training_set:
                counts = instance.get_word_freq()
                if word in counts:
                    word_freq += 1

            self.freq_weights[word] = m.log(instance_number / word_freq)

        return

    def train(self, training_set, words_to_use):
        self.set_frequency_weights(training_set, words_to_use)
        NaiveBayesClassifier.train(self, training_set, words_to_use)
        return


class TransformedComplementNaiveBayesClassifier(DocumentTransformedComplementNaiveBayesClassifier):
    """
    Complement Bayes with Document Frequency and Logarithmic Term frequency transformation
    """
    def __repr__(self):
        return "TF-IDF transformed Complement Naive Bayes"

    def transform(self, freq, word):
        freq = m.log(1 + freq) * self.freq_weights[word]
        return freq 
    






    