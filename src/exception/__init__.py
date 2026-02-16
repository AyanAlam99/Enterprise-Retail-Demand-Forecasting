import sys 
import logging

def error_message_detail(error:Exception , error_detail :sys)->str : 
    """
    Extracts detailed error info including file naem , line numb , and the error mssg 
    
    :param error: The exception that occured
    :param error_detail: The sys module to access traceback details 
    :return: A formatted error mssg string 
    """

    _,_,exec_tb= error_detail.exc_info()
    file_name = exec_tb.tb_frame.f_code.co_filename

    line_numb = exec_tb.tb_lineno

    error_message = f"Error occured in python script :[{file_name}] at line number [{line_numb}] : {str(error)}"

    logging.error(error_message)

    return error_message


class MyException(Exception): 
    
    def __init__(self , error_message : str, error_detail : sys) : 

        super().__init__(error_message)

        self.error_message = error_message_detail(error_message,error_detail)

    def __str__(self)->str : 
        """
        returns the string reprsntn of the error mssg
        """
        return self.error_message