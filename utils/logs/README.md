Logging Utility
---

This came about me messing about creating a game interface with the `logging` module and is therefore **far** more complex then required but it's really fancy so I use it everywhere now.

# Usage

Inside each module declare

```
import logs
logs.getHandlers(filename='logfile.log', mode='w+')
logger = logs.get_logger('identifier')
```

Then use logging as normal and receive the output with the module and line number that the log was created from

```
 module: linenumber: type: message
```

Example in the `ipython` shell

```
import logs
logs.getHandlers(filename='logfile.log', mode='w+')
logger = logs.get_logger('identifier')
logger.info('test')
```

to give

```
   logs:    74: Info  : Logger started: identifier
 <ipython-input-1-56db9d1f0dbb>:     4: Info  : test
```

and a log entry in `logfile.log` as

```
logs:  logs.logs:    59:   DEBUG : Handlers added
logs: identifier:    74:    INFO : Logger started: identifier
 <ipython-input-1-56db9d1f0dbb>: identifier:     4:    INFO : test
```

## Log a Class
Logging within a class can identify the class in operations by adding `self` as a second argument

    import logs
    logs.getHandlers(filename='logfile.log', mode='w+')
    
    class My_Class():
        def __init__(self):
            logs.get_logger(self)
            self.logger.info('a logger is attached to `self` by `logs.get_logger`')
        
    m = My_Class()

which displays as normal

```
       logs:    70: Info  : Logger started: identifier
 <ipython-input-8-1e343d7d8015>:     7: Info  : a logger is attached to `self` by `logs.get_logger`
```

and a log entry in `logfile.log` displays further detail as

```
       logs:  logs.logs:    55:   DEBUG : Handler already added
       logs:  logs.logs:    55:   DEBUG : Handler already added
       logs: identifier:    70:    INFO : Logger started: identifier
 <ipython-input-8-1e343d7d8015>: identifier:     7:    INFO : a logger is attached to `self` by `logs.get_logger`
```

# Configuration
Global configuration should be done in the file

    logs.py

for example

    logging.root.setLevel(logging.DEBUG)

displays all messages including debugging

See the [`logging` docs](https://docs.python.org/2/howto/logging.html#when-to-use-logging) for more information

## Customisation
It is possible with this set-up to define custom severity levels, respective default message formats and also colourings in the terminal window.

### Severity Levels
In addition to [regular states](https://docs.python.org/2/howto/logging.html#when-to-use-logging), there is the option for custom `logging` states. Currently these are set as

    logging.l1('example message')
    logging.l2('example message')
    logging.l1('example message')

with severity levels

    logging.L1
    logging.L2
    logging.L3

the set up of this is particularly tricky and so to define a custom logging name. For example to achieve

    logging.root.setLevel(logging.OMG)
    logging.omg('example message')

the simplest way is to search and replace all instances of `L1,l1` with `OMG, omg` in the file

    ./logging_formatter.py

### Severity Colourings
To define the colours, edit the top part of

    logging_formatter.py

Changing the colours just requires assigning new colour names to the following
    
    # This will change the colour in the run-time messages
    COLOR_CONFIG = {
        "L1":CYAN,
        "L2":WHITE,
        "L3":YELLOW
    }

The colour names are hardcoded to match the definitions in

    logging_colourer.py

which is very messy for good reason so don't poke around in this file or change the lines

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
    
unless you know what you're doing!

#### Default Severity Colourings
If you want to *poke* around and change the default colourings then this can be done in

    logging_colourer.py

by changing the following lines with new colour names

    # add methods we need to the class
    def new(*args):
        levelno = args[1].levelno
        if(levelno>=50):
            color = red
        elif(levelno>=40):
            color = red
        elif(levelno>=30):
            color = yellow
        # These are the custom logging levels that I use
        elif(levelno>=27):
            color = plain
        elif(levelno>=26):
            color = red
        elif(levelno>=25):
            color = pink
        elif(levelno>=24):
            color = green
        elif(levelno>=23):
            color = yellow
        elif(levelno>=22):
            color = cyan
        elif(levelno>=21):
            color = white
        # The end of the custom logging levels that I use
        elif(levelno>=20):
            color = green
        elif(levelno>=10):
            color = pink
        else:
            color = plain
        
        args[1].msg = color + str(args[1].msg) + plain # normal
        #print "after"
        return fn(*args)