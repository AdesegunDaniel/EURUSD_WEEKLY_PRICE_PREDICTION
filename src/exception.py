import sys
import logging

def error_message_detail(error, error_details:sys):
    _,_,exc_tb=error_details.exc_info()
    file_name=exc_tb.tb_frame.f_code.co_filename
    error_message="hold on a munite! is like you make a mistake in {0} line {1} [{2}]]".format(file_name, exc_tb.tb_lineno, str(error))
    return error_message
class CustomException(Exception):
    def __init__(self, error_message,error_detail):
        super().__init__(error_message)
        self.error_message=error_message_detail(error_message, error_detail)

    def __str__(self):
        return self.error_message
    