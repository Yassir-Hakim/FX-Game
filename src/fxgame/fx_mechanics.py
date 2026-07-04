from dataclasses import dataclass


# Default parameters of the base game. 
A0_DEFAULT = 1.25       # last revealed rate (dollars per pound)
sd_DEFAULT = 0.05    # standard deviation of one random-walk step
BDC_FEE_DEFAULT = 0.02  # the trader's Bureau de Change fee (2%)


def mm_accepts(offer_rate: float, true_rate: float):
    #The market maker's accept/reject rule for a POUND-SELLER.
    
    return offer_rate < true_rate


def bdc_payoff_per_pound(true_rate: float, fee: float = BDC_FEE_DEFAULT):
    #Dollars per pound from selling pounds at the Bureau de Change.
    
    return (1.0 - fee) * true_rate


@dataclass
class GameParams:
    #The parameters that define one instance of the base game.

    a0: float = A0_DEFAULT
    sd: float = sd_DEFAULT
    bdc_fee: float = BDC_FEE_DEFAULT
 
