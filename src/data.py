from .instances import TrainingSet, TrainingInstance, TestSet, TestingInstance

class CSVHandler:
    """
    Class for handling reading of training and testing csv files as well
    as writing to the ouput predictions csv files
    """
    def __init__(self, train_filename, test_filename, output_filename):
        self.train_filename = train_filename
        self.test_filename = test_filename
        self.output_filename = output_filename

    def read_training_file(self):
        """
        Read training csv file into a TrainingSet object
        """
        training_set = TrainingSet()
        with open(self.train_filename, 'r') as training_file:
            lines_lst = training_file.readlines()
            lines_lst.pop(0)  # remove header line

            for line in lines_lst:
                instance_id, instance_label, instance_text = line.split(",") # Note training instance has label
                instance = TrainingInstance(int(instance_id), instance_label, instance_text[1:-2])
                training_set.add_instance(instance)

        return training_set

    def read_testing_file(self):
        """
        Read testing csv file into a TestSet object
        """
        test_set = TestSet()
        with open(self.test_filename, 'r') as testing_file:
            lines_lst = testing_file.readlines()
            lines_lst.pop(0)  # remove header line

            for line in lines_lst:
                instance_id, instance_text = line.split(",")
                instance = TestingInstance(int(instance_id), instance_text[1:-2])
                test_set.add_instance(instance)

        return test_set

    def write_output(self, predictions):
        """
        Given a list of predictions in (id, prediction) form, write these to a csv file
        """
        with open(self.output_filename, 'w') as output_file:
            output_file.write("id,class\n") # Header
            for prediction in predictions:
                instance_id, predicted_label = prediction
                output_line = "{},{}\n".format(instance_id, predicted_label)
                output_file.write(output_line)

        return



    