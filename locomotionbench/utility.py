import pandas as pd
import numpy as np


def settings_to_dict():
    return True


class PIExporter:

    @staticmethod
    def write_file(filename, file_content):
        try:
            with open(filename, 'w') as file:
                file.write(file_content)

            file.close()
        except:
            print("Failed to write output to file \"{}\"".format(filename))
            raise

    @staticmethod
    def input_to_string(input_values):
        try:
            np_input = np.array(input_values)
        except:
            try:
                np_input = input_values.to_numpy()
            except:
                print("Failed to convert input to numpy array!")
                raise

        try:
            out = np.array2string(np_input, threshold=np.inf, max_line_width=np.inf, separator=', ').replace('\n', '')
        except:
            print("Failed to convert numpy_input to string!")
            raise

        return out

    def export_string(self, values, filename=None):
        file_out = '''type: \'string\'\nvalue: \'{}\'\n'''.format(values)

        if filename is not None:
            self.write_file(filename, file_out)

        return file_out

    def export_scalar(self, values, filename=None):
        file_out = '''type: \'scalar\'\nvalue: {}\n'''.format(values)

        if filename is not None:
            self.write_file(filename, file_out)

        return file_out

    def export_vector(self, values, filename=None, labels=None):
        np_vec_str = self.input_to_string(values)

        labels_out = ""

        if labels is not None:
            if len(values) != len(labels):
                raise ValueError("Labels count doesn't match vector length!")
            labels_out = '\nlabel: {}'.format(labels)

        file_out = '''type: \'vector\'{}\nvalue: {}\n'''.format(labels_out, np_vec_str)

        if filename is not None:
            self.write_file(filename, file_out)

        return file_out

    def export_matrix(self, values, filename=None, row_labels=None, col_labels=None):

        np_vec_str = self.input_to_string(values)

        row_labels_out = ""
        col_labels_out = ""
        aggregations_out = ""

        if row_labels is not None:
            row_labels_out = '\nrow_label: {}'.format(row_labels)

        if col_labels is not None:
            col_labels_out = '\ncol_label: {}'.format(col_labels)

        file_out = '''type: \'matrix\'{}{}\nvalue: {}\n'''.format(row_labels_out, col_labels_out, np_vec_str)

        if filename is not None:
            self.write_file(filename, file_out)

        return file_out
