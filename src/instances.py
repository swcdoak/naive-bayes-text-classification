
class AbstractInstance:
    """
    Parent instance class to encapsulate common methods and attributes
    to both TrainingInstance and TestingInstance
    """
    def __init__(self, instance_id, instance_text):
        self.id = instance_id
        self.text = instance_text
        
        word_freq = self.count_words(instance_text) # Get frequencies of words in abstract
        self.word_freq = word_freq

    def get_text(self):
        return self.text

    def get_word_freq(self):
        return self.word_freq

    @staticmethod
    def count_words(text):
        """
        Constructs a dictionary of (word, frequency) pairs for the instances
        abstract
        """
        text_lst = text.split(" ")
        word_freq = {word: 0 for word in text_lst}

        for word in text_lst:
            word_freq[word] += 1

        return word_freq

    def update_counts(self, words_to_keep):
        """
        words_to_keep - Most common words chosen in feature selection
        
        Part of preprocessing step, once we have performed feature selection
        (found the n most common words amoung all abstracts) we remove the unused
        features (words) from the dictionaries
        """
        new_freq = {}

        for word in words_to_keep:
            if word in self.word_freq:
                new_freq[word] = self.word_freq[word]

        self.word_freq = new_freq

        return


class TrainingInstance(AbstractInstance):
    """
    Class to representing Training Instance. The primary difference between
    TrainingInstance and TestingInstance being TrainingInstance's label attribute
    """
    def __init__(self, instance_id, instance_label, instance_text):
        AbstractInstance.__init__(self, instance_id, instance_text)
        self.label = instance_label

    def __repr__(self):
        return "TrainingInstance: id={}, class={}".format(self.id, self.label)

    def copy(self):
        """
        Returns copy of instance, word_freq attribute is copied deeply while others
        are shallowly copied
        """
        copy_instance = TrainingInstance(self.id, self.label, self.text)
        copy_instance.word_freq = self.word_freq.copy()
        return copy_instance


class TestingInstance(AbstractInstance):
    """
    For readability
    """
    def __repr__(self):
        return "TestingInstance id={}".format(self.id)


class AbstractInstanceSet:
    """
    Parent instance set class to encapsulate common attributes/methods
    InstanceSet classes hold a list of their respective type of instance objects
    with some usefule methods
    """
    def __init__(self):
        self.instances = []

    def __iter__(self):
        """
        Allows iteration through instance set using the same interface as
        the class attribute 'instances' list.
        """
        return iter(self.instances)

    def __getitem__(self, item):
        return self.instances[item]

    def __len__(self):
        return len(self.instances)

    def add_instance(self, instance):
        self.instances.append(instance)
        return


class TrainingSet(AbstractInstanceSet):
    def __add__(self, other):
        """
        Add TrainingSets by adding instances attributes
        """
        new_set = TrainingSet()
        new_set_lst = self.instances + other.instances
        new_set.instances = new_set_lst
        return new_set
    
    def copy(self):
        copy_set = TrainingSet()
        for instance in self.instances:
            copy_set.add_instance(instance.copy())
        return copy_set

    def separate_by_class(self):
        """
        Returns a dictionary of (label, TrainingSet) pairs where the instances have
        been organized into their respective classes
        """
        classes = ["A", "B", "E", "V"]
        separation = {label:TrainingSet() for label in classes}

        for instance in self.instances:
            instance_label = instance.label
            separation[instance_label].add_instance(instance)

        return separation

    def get_all_counts(self):
        """
        Returns Dictionary containing (word, frequency) pairs across entire training set
        """
        all_counts = {}

        for instance in self.instances:
            instance_counts = instance.get_word_freq()

            for word in instance_counts:
                if word in all_counts:
                    all_counts[word] += instance_counts[word]

                else:
                    all_counts[word] = instance_counts[word]

        return all_counts


class TestSet(AbstractInstanceSet):
    """
    For readability
    """
    pass


