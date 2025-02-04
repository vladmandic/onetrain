# OneTrain: Single-stop for training LoRA models

1. Validate dataset: sizes, dynamic ranges, faces, etc.
2. Prepare dataset: convert, optimize buckets, etc.
3. Caption dataset: multiple captioning models
4. Train model: using [OneTrainer](https://github.com/Nerogar/OneTrainer/)
5. Unified logging and real-time monitoring

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

## Examples

See `examples/` folder for both **SD-XL** and **FLUX.1-dev** LoRA training scripts and configs  

## Reference

- <https://github.com/Nerogar/OneTrainer/wiki/LoRA>
- <https://github.com/Nerogar/OneTrainer/wiki/Training>
- <https://github.com/Nerogar/OneTrainer/wiki/Optimizers>
- <https://civitai.com/articles/3105/essential-to-advanced-guide-to-training-a-lora>
