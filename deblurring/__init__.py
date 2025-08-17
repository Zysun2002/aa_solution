from .deblur_probability_based import deblur
from .overlap_for_confidence import overlap4conf
# from .overlap_for_mask import overlap4mask
from .extract_from_exp import extract as extract_exp
from .extract_from_exp import latest_exp_folder

import ipdb

def triCls_debl(data_path, exp_path):
    # ipdb.set_trace()
    print("deblurring ...")    
    latest_exp_path = latest_exp_folder(exp_path, "val")
    
    # do overlapping in exp folder
    overlap4conf(latest_exp_path)

    # copy overlapping result to data folder
    extract_exp(latest_exp_path, data_path)

    # deblur
    deblur(data_path/"val")


def ddf_debl(data_path, exp_path):
    print("deblurring ...")    
    latest_exp_path = latest_exp_folder(exp_path)
    # overlap4mask(latest_exp_path)
    # overlap4conf(latest_exp_path)
    
    # overlap_ddf()
    extract_exp(latest_exp_path, data_path)
    # deblur(data_path/"val")