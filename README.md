# OneTrainer Wrapper

Wrapper for amazing [OneTrainer](https://github.com/Nerogar/OneTrainer) by [@nerogar](https://github.com/nerogar)  
With uniform logging and progress reporting  
Can be used as a *command line tool* or as a *Python module*  

## Notes

All configuration defaults are stored in `templates.py` and common ones can be overriden using args  
Default configuration is for LoRA training for SDXL with 100 epochs using *scheduler:cosine* and *optimizer:prodigy*  

Mandatory params are only `concept` and `input`  
Plus command do you want to run `caption` or `train` or both  

If you want to use different starting defaults, you can specify config json file using `config` flag  

`onetrain.py` performs python import of actual **OneTrainer** modules,  
so if they are not in the same folder, specify `onetrainer` path  
All operations are native, there are no shell executes  

Temp location if not specified is `SYSTEM_TEMP_FOLDER/onetrain`  
Log if not specified will be stored in `SYSTEM_TEMP_FOLDER/onetrain/onetrain.log`  

### Training

- Creates OneTrainer config files in temp location using template and with specified overrides
- If output location is not selected, it will also be in temp location
- Kicks off training
- Maintains `info` object with all updates about training in progress  
- Cleans up config metadata and stores it in the output model  

### Captioning

- Creates txt files next to each image file in input dataset
- Adds concept name as initial tag
- Appends general description using BLIP2 model
- Appends detailed tags using WD14 model

## Usage: CLI

> python onetrain.py --help

    options:
      --concept CONCEPT           concept name
      --input INPUT               folder with training dataset
      --model MODEL               stable diffusion base model
      --train                     run training
      --caption                   run captioning
      --onetrainer ONETRAINER     path to onetrainer
      --author AUTHOR             train author to be included in metadata
      --resume                    resume training from last backup
      --config CONFIG             use specific onetrainer config file
      --type {sd,sdxl}            model type
      --log LOG                   specify log file
      --output OUTPUT             specify output location
      --epochs EPOCHS             number of training epochs
      --accumulation ACCUMULATION gradient accumulation steps
      --optimizer OPTIMIZER       training optiomizer
      --scheduler SCHEDULER       training scheduler
      --rank RANK                 lora rank
      --alpha ALPHA               lora alpha
      --batch BATCH               training batch size
      --backup BACKUP             create n training backups
      --save SAVE                 save n intermittent steps
      --resolution RESOLUTION     training override resolution
      --bias                      use bias correction
      --triton                    use triton
      --te                        train text encoder
      --tmp TMP                   training temp folder
      --nopbar                    disable progress bar

### Example

> python onetrai.py --caption --train --model ./sdxl.safetensors --concept alla --input ./dataset --author vladmandic --batch 4 --onetrainer ~/code/onetrainer --epoch 200

    18:13:16-346173 INFO     onetrain
    18:13:16-347604 INFO     log: /tmp/onetrainer/onetrain.log
    18:13:16-348139 INFO     args: ...
    18:13:18-818323 INFO     caption: model=blip2 path=/home/vlado/dev/onetrainer/dataset
    caption blip ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 100% 0:00:04 0:00:00 21/21 images
    18:22:58-622108 INFO     caption: model=wd14 path=/home/vlado/dev/onetrainer/dataset
    caption wd14 ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 100% 0:00:09 0:00:00 21/21 images
    18:23:22-665608 INFO     concept: alla samples=21 path=/home/vlado/dev/onetrainer/dataset buckets={'1024x934': 1, '1024x820': 4, '1024x846': 1, '1024x819': 3, '1024x913': 1, '1024x823': 2, '1024x821': 2, '1024x826': 1, '1024x576': 1, '1024x924': 1, '1024x866': 1, '1024x949': 1, '1024x909': 1, '1024x1024': 1}
    18:23:23-076770 INFO     config: file=/tmp/onetrainer/config.json
    18:23:23-077715 INFO     concepts: name=alla file=/tmp/onetrainer/concept.json
    18:23:23-079219 INFO     samples: file=/tmp/onetrainer/samples.json
    18:23:23-079914 INFO     method=LORA type=STABLE_DIFFUSION_XL_10_BASE
    18:23:23-080537 INFO     model=sdxl.safetensors
    18:23:23-081757 INFO     train: init
    18:23:23-082280 INFO     update: loading the model
    18:23:25-700114 INFO     update: running model setup
    18:23:28-127638 INFO     update: creating the data loader/caching
    18:23:28-129684 INFO     optimizer=PRODIGY scheduler=COSINE rank=32 alpha=16.0 batch=4 accumulation=1 epochs=200
    18:23:28-130432 INFO     train: start
    18:25:38-805663 INFO     update: saving
    train ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 100% 0:20:42 0:00:00 step: 800 epoch: 201/200 batch: 0 samples: 4 its: 0.64
    18:44:06-031347 INFO     update: saving the final model
    18:44:06-748128 INFO     save: /tmp/onetrainer/alla.safetensors
    18:44:06-748807 INFO     train: completed

## Usage: Module

```py
import onetrain
onetrain.args.epoch = 100
onetrain.prepare(onetrain.args)
onetrain.caption(onetrain.args)
onetrain.train(onetrain.args)
onetrainer.log(onetrain.info) # info object contains real-time information about training that can be monitored
```

or if you want to use it async so your app can do other things and monitor progress

```py
import threading
import onetrain
onetrain.args.epoch = 100
onetrain.prepare(onetrain.args)
onetrain.caption(onetrain.args)
thread = threading.Thread(target=onetrain.train, args=(onetrain.args,))
thread.start()
while thread.is_alive():
    onetrain.log(onetrain.info)
    time.sleep(10)
```
