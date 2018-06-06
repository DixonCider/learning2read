# make track1, track2 submissions
import numpy as np

class Track1:
    """
        'class'  : 'learning2read.submission.Track1',
        'output' : 'track1',
        'input_data' : ['df_test', 'model_rf'],
    """
    @classmethod
    def predict(cls,df_test,model):
        df_test = df_test.drop('Book-Rating', 1)
        return list(model.predict(df_test))

    @classmethod
    def check_input(cls,input_data):
        assert type(input_data)==list
        assert len(input_data)==2

    @classmethod
    def run(cls,input_data):
        cls.check_input(input_data)
        y_pred = cls.predict(*input_data)
        return {
            'output' : [min(10,max(1,int(round(y)))) for y in y_pred]
        }


class Track2(Track1):
    @classmethod
    def run(cls,input_data):
        cls.check_input(input_data)
        y_pred = cls.predict(*input_data)
        # round because file size limit ... QAQ
        return {
            'output' : [min(10,max(1,round(y,1))) for y in y_pred]
        }
        