import configfile
import wandb
import train
import icecream as ic
import pysnooper
if  __name__ == '__main__':
  wandb.login()
  # tell wandb to get started
  with wandb.init(project="pytorch-demo", config=configfile.callconfig()):
    print(type(configfile.callconfig()))
    # access all HPs through wandb.config, so logging matches execution!
    config=wandb.config
    #print(dir(config))
    # make the model, data, and optimization problem
    # and use them to train the model
    train.train(config) 
        # and test its final performance
