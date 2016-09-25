#!/usr/bin/env python
# encoding: utf-8

# directory-wide and shared processes
import sys
import logging
from logging_colourer import * # created pretty colors for logger
from logging_formatter import * # created pretty colors for logger
from functools import wraps

# http://stackoverflow.com/a/6307868/4013571
def wrap_all(decorator):
    """wraps all function with the wrapper provided as an argument"""
    def decorate(cls):
        for attr in cls.__dict__: # there's propably a better way to do this
            if callable(getattr(cls, attr)):
                setattr(cls, attr, decorator(getattr(cls, attr)))
        return cls
    return decorate

def log_me(func):
    """Adds a logging facility to all functions it wraps"""
    @wraps(func)
    def tmp(*args, **kwargs):
        func_name = func.__name__
        if func_name != '__init__':
            args[0].logger.debug('...running {}'.format(func_name))
        return func(*args, **kwargs)
    return tmp

def getHandlers(filename=None, mode='w+', file_level=logging.DEBUG, stream_level=logging.INFO):
    """Prints to file. Sets up the handlers only once"""
    
    #   create stream handler
    create_new_levels(COLOR_CONFIG)
    stream_fmt = MyFormatter()
    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(stream_fmt)
    stream_handler.setLevel(stream_level)
    
    #   create file handler
    file_handler = logging.FileHandler(filename=filename, mode=mode)
    file_handler.setLevel(file_level)
    file_fmt = logging.Formatter(" %(module)10s: %(name)10s: %(lineno)5d: %(levelname)7s : %(msg)s")
    file_handler.setFormatter(file_fmt)
    
    if len(logger.root.handlers) > 0:
        for handler in logger.root.handlers:
            # add the handlers to the logger
            # makes sure no duplicate handlers are added
            if not isinstance(handler, logging.FileHandler) and not isinstance(handler, logging.StreamHandler):
                logging.root.addHandler(stream_handler)
                if filename is not None: logging.root.addHandler(file_handler)
            else:
                logger.debug('Handler already added')
    else:
        logging.root.addHandler(stream_handler)
        if filename is not None: logging.root.addHandler(file_handler)
        logger.debug('Handlers added')
    pass

def get_logger(name=None, self=None):
    """Makes a logger based on the context"""
    # creates a logger for the test file
    
    if self is not None:
        if name is None: name = type(self).__name__
        logger = logging.getLogger(name)
        self.logger = logger
        self.logger.info('Logger started: {}'.format(name))
    else:
        if name is None: raise ValueError('Need name defined if self is None')
        logger = logging.getLogger(name)
        logger.info('Logger started: {}'.format(name))
    return logger

# This should be left as DEBUG
# change logging level with getHandlers
logging.root.setLevel(logging.DEBUG)
logger = logging.getLogger(__name__)