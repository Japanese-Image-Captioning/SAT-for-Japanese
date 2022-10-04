from dataclasses import dataclass
from datasets import *
from utils import *
from pycocoevalcap.bleu.bleu import Bleu
from pycocoevalcap.meteor.meteor import Meteor
from pycocoevalcap.rouge.rouge import Rouge
from pycocoevalcap.cider.cider import Cider
from pycocoevalcap.spice.spice import Spice


@dataclass
class Metrics:
    bleu: float
    rouge: float 
    meteor: float 
    cider: float 
    spice: float

def compute_metrics(references, candidates,is_ja):
    ###BLEU#####
    print("Compute BLEU ... ")
    pycoco_bleu = Bleu()
    bleu, _ = pycoco_bleu.compute_score(references, candidates)

    ####METEOR###
    print("Compute METEOR ... ")
    pycoco_meteor = Meteor()
    meteor, _ = pycoco_meteor.compute_score(references, candidates)
    del pycoco_meteor
    # meteor = 0 # METEORはたまにバグるので

    ####ROUGE###
    print("Compute ROUGE ... ")
    pycoco_rouge = Rouge()
    rouge, _ = pycoco_rouge.compute_score(references, candidates)

    ####CIDER###
    print("Compute CIDER ... ")
    pycoco_cider = Cider()
    cider, _ = pycoco_cider.compute_score(references, candidates)

    ####SPICE####
    print("Compute SPICE ... ")
    if is_ja:
        spice = 0
    else:
        pycoco_spice = Spice()
        spice, _ = pycoco_spice.compute_score(references, candidates)

    metrics = Metrics(bleu, rouge, meteor, cider, spice)

    return metrics
