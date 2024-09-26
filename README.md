# OneTrainer Wrapper

Wrapper for [OneTrainer](https://github.com/Nerogar/OneTrainer) by [@nerogar](https://github.com/nerogar)  
With image validation, captioning, logging and progress reporting  
Can be used as a *command line tool* or as a *Python module*  

## Usage: CLI

> python onetrain.py --help

## Usage: Module

```py
import onetrain
onetrain.args.epoch = 100
onetrain.prepare(onetrain.args)
onetrain.caption(onetrain.args)
onetrain.train(onetrain.args)
onetrainer.log(onetrain.info) # info object contains real-time information about training that can be monitored
```
