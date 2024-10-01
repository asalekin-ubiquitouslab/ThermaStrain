from models.baseline import Baseline,multitask
from models.multimodal import  MM

from models.propose import P2
from models.lossfunction import flow2KL,flow2cmd,flow2mmd
from models.features import feature

Models = {
   
    'baseline':Baseline,
    'mm':MM,
    'p2':P2,
    'kl':flow2KL,
    'cmd':flow2cmd,
    'mmd':flow2mmd,
    'multitask':multitask,
    'feature':feature,
    

}
#baseline
#multimodal
#halluci