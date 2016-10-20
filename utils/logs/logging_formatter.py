#!/usr/bin/env python
# encoding: utf-8
import logging
import types
import sys

# THESE ARE HERE 
# FOR ASSIGNING COLORS
WHITE   = 21
CYAN    = 22
YELLOW  = 23
GREEN   = 24
PINK    = 25
RED     = 26
PLAIN   = 27
# USED IN COLOR_CONFIG

# This will change the colour in the run-time messages
COLOR_CONFIG = {
    "L1":CYAN,
    "L2":WHITE,
    "L3":YELLOW
}

# Custom formatter for logging
# http://stackoverflow.com/a/8349076/4013571
class MyFormatter(logging.Formatter):
    """Class to create custom formats"""
    
    # these lines seem a bit awkward but I don't know 
    # enought python to change them!
    # exmpl_fmt = "WARN:  %(module)s: %(name)s: %(lineno)d: %(msg)s"
    dbg_fmt     = " %(module)10s: %(lineno)5d: Debug : %(msg)s"
    l1_fmt      = " %(module)10s: %(lineno)5d: L1    : %(msg)s"
    l2_fmt      = " %(module)10s: %(lineno)5d: L2    : %(msg)s"
    l3_fmt      = " %(module)10s: %(lineno)5d: L3    : %(msg)s"
    info_fmt    = " %(module)10s: %(lineno)5d: Info  : %(msg)s"
    wrn_fmt     = " %(module)10s: %(lineno)5d: Warn  : %(msg)s"
    err_fmt     = " %(module)10s: %(lineno)5d: Error : %(msg)s"
    
    def __init__(self, fmt="%(levelno)s: %(msg)s"):
        logging.Formatter.__init__(self, fmt)
    
    def format(self, record):
        
        # Save the original format configured by the l1
        # when the logger formatter was instantiated
        format_orig = self._fmt
        
        # Replace the original format with one customized by logging level
        if record.levelno == logging.DEBUG:
            self._fmt = MyFormatter.dbg_fmt
        elif record.levelno == logging.INFO:
            self._fmt = MyFormatter.info_fmt
        elif record.levelno == logging.L1:      # This is for the L1 messages
            self._fmt = MyFormatter.l1_fmt
        elif record.levelno == logging.L2:      # This is for the l2 messages
            self._fmt = MyFormatter.l2_fmt
        elif record.levelno == logging.L3:      # This is for the l3 messages
            self._fmt = MyFormatter.l3_fmt
        elif record.levelno >= logging.WARNING:
            self._fmt = MyFormatter.wrn_fmt
        elif record.levelno >= logging.ERROR:
            self._fmt = MyFormatter.err_fmt
        
        # Call the original formatter class to do the grunt work
        result = logging.Formatter.format(self, record)
        
        # Restore the original format configured by the l1
        self._fmt = format_orig
        
        return result


def create_new_levels(COLOR_CONFIG):
    ### Create new Logging level for game messages ###
    # set up custom logging level or game output
    L1_COLOR = COLOR_CONFIG["L1"]
    L2_COLOR = COLOR_CONFIG["L2"]
    L3_COLOR = COLOR_CONFIG["L3"]
    
    L2 = L2_COLOR
    logging.addLevelName(L2, "L2")
    logging.L2 = L2
    def l2(self, message, *args, **kwargs):
        # Yes, logger takes its '*args' as 'args'
        if self.isEnabledFor(L2):
            self._log(L2, message, args, **kwargs)
    logging.Logger.l2 = l2
    ### End creation of new logging level ####
    
    ### Create new Logging level for game messages ###
    # set up custom logging level or game output
    L1 = L1_COLOR
    logging.addLevelName(L1, "L1")
    logging.L1 = L1
    def l1(self, message, *args, **kwargs):
        # Yes, logger takes its '*args' as 'args'
        if self.isEnabledFor(L1):
            self._log(L1, message, args, **kwargs)
    logging.Logger.l1 = l1
    ### End creation of new logging level ####
    
    ### Create new Logging level for L1 messages ###
    # set up custom logging level or l3 output
    L3 = L3_COLOR
    logging.addLevelName(L3, "L3")
    logging.L3 = L3
    def l3(self, message, *args, **kwargs):
        # Yes, logger takes its '*args' as 'args'
        if self.isEnabledFor(L3):
            self._log(L3, message, args, **kwargs)
    logging.Logger.l3 = l3
    ## End creation of new logging level ####
    pass