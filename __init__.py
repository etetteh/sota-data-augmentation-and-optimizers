#data aug
from augmentation.autoaugment import CIFAR10Policy
from augmentation.cutout import Cutout
from augmentation.AugMix import AugMixDataset
from augmentation.RandAugment import RandAugment

#optim and activation
from optim.deepmemory   import DeepMemory
from optim.lookahead    import Lookahead
from optim.radam        import RAdam

from loss_func.cross_entropy import CrossEntropyLoss
from models.model import CustomModel
from models.model_isda import CustomModel, Full_layer

