
from .instances import TrainingInstance, TrainingSet, TestingInstance, TestSet

class Preprocessor:
    """
    Class for preprocessing training set, specifically:
        1.) Find the most common words
            -> If desired remove stop words
        2.) Remove unused words from training instance dictionaries, making them ready for training
        3.) Process test instances similarly using the same common words
    """
    def __init__(self):
        self.most_frequent_words = []
        self.useless_words = ["the", "of", "and", "a", "in", "to", "that", "is", "with", "for", "from", "are", "was",
                              "by", "were", "as", "this", "which", "an", "we", "have", "these", "has", "be", "been",
                              "other", "on", "at", "or", "but"]

    def get_most_frequent_words(self):
        return self.most_frequent_words

    def set_most_frequent_words(self, training_set, n=1000, remove_useless=False):
        """
        Find the most n frequent words across all training instances. When remove_useless is True:
        Remove the stop words.
        """
        super_count = training_set.get_all_counts() # Combined frequency dictionary of all words in training set

        class CountComparator:
            """
            Helper class allowing us to sort (word, frequency) pairs based on their frequency
            """
            def __init__(self, word, count):
                self.word = word
                self.count = count

            def __lt__(self, other):
                return self.count < other.count

        ranked_comps = []
        for word in super_count:
            comp = CountComparator(word, super_count[word])
            ranked_comps.append(comp)

        ranked_comps.sort(reverse=True)
        ranked_words = [comp.word for comp in ranked_comps]

        if remove_useless:
            # If desired, remove stop words
            new_ranked_words = []
            for word in ranked_words[:100]:
                # Useless were words handpicked from top 100
                if word not in self.useless_words:
                    new_ranked_words.append(word)

            ranked_words = new_ranked_words + ranked_words[100:]

        self.most_frequent_words = ranked_words[:n] # Only take top n words

        return self.most_frequent_words

    def process_training_set(self, training_set):
        """
        Removes all unused words from the training instance's dictionaries
        """
        processed_training_set = TrainingSet()

        for instance in training_set:
            # Changed stuff here
            processed_instance = instance.copy()
            processed_instance.update_counts(self.most_frequent_words)
            processed_training_set.add_instance(processed_instance)

        return processed_training_set

    def process_test_set(self, test_set):
        processed_test_set = TestSet()

        for instance in test_set:
            instance_id = instance.id
            instance_text = instance.text
            
            # Make a deep copy of processed test set
            processed_instance = TestingInstance(instance_id, instance_text)
            processed_instance.update_counts(self.most_frequent_words)
            processed_test_set.add_instance(processed_instance)

        return processed_test_set


