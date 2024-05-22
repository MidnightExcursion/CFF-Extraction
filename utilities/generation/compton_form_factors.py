from numpy import power, exp

def generic_compton_form_factor_function(
        x_Bjorken: float = 0., 
        t_hadron_momentum_transfer: float = 0., 
        parameter_a: float = 0., 
        parameter_b: float = 0., 
        parameter_c: float = 0., 
        parameter_d: float = 0., 
        parameter_e: float = 0., 
        parameter_f: float = 0.):
    
    try:

        # (1): Calculate the Polynomial Part:
        polynomial_part = (parameter_a * (power(x_Bjorken, 2)) + parameter_b * x_Bjorken)

        # (2): Calculate the Exponential Part:
        exponential_part = exp(parameter_c * (power(t_hadron_momentum_transfer, 2) + parameter_d * t_hadron_momentum_transfer + parameter_e))

        # (3): Put it together: Poly * Exp + Const:
        example_cff = polynomial_part * exponential_part + parameter_f

        # (4): Return the basic model of the CFF:
        return example_cff
    
    except Exception as ERROR:

        print(f"> Error in generating CFF input for xB = {x_Bjorken} and t = {t_hadron_momentum_transfer}:\n {ERROR}")
        return 0.
    
def cff_Re_H(
    x_Bjorken: float,
    t_squared_hadronic_momentum_transfer: float) -> float:
    """
    CFF: Re[H]
    """

    _PARAMETER_A = -4.41
    _PARAMETER_B = 1.68
    _PARAMETER_C = -9.14
    _PARAMETER_D = -3.57
    _PARAMETER_E = 1.54
    _PARAMETER_F = -1.37

    return generic_compton_form_factor_function(
        x_Bjorken,
        t_squared_hadronic_momentum_transfer,
        _PARAMETER_A,
        _PARAMETER_B,
        _PARAMETER_C,
        _PARAMETER_D,
        _PARAMETER_E,
        _PARAMETER_F)

def cff_Re_E(
    x_Bjorken: float,
    t_squared_hadronic_momentum_transfer: float) -> float:
    """
    CFF: Re[E]
    """

    _PARAMETER_A = 144.56
    _PARAMETER_B = 149.99
    _PARAMETER_C = 0.32
    _PARAMETER_D = -1.09
    _PARAMETER_E = -148.49
    _PARAMETER_F = -0.31

    return generic_compton_form_factor_function(
        x_Bjorken,
        t_squared_hadronic_momentum_transfer,
        _PARAMETER_A,
        _PARAMETER_B,
        _PARAMETER_C,
        _PARAMETER_D,
        _PARAMETER_E,
        _PARAMETER_F)

def cff_Re_He(
    x_Bjorken: float,
    t_squared_hadronic_momentum_transfer: float) -> float:
    """
    CFF: Re[He]
    """

    _PARAMETER_A = -1.86
    _PARAMETER_B = 1.50
    _PARAMETER_C = -0.29
    _PARAMETER_D = -1.33
    _PARAMETER_E = 0.46
    _PARAMETER_F = -0.98

    return generic_compton_form_factor_function(
        x_Bjorken,
        t_squared_hadronic_momentum_transfer,
        _PARAMETER_A,
        _PARAMETER_B,
        _PARAMETER_C,
        _PARAMETER_D,
        _PARAMETER_E,
        _PARAMETER_F)

def cff_DVCS(
    x_Bjorken: float,
    t_squared_hadronic_momentum_transfer: float) -> float:
    """
    Deeply Virtual Compton Scattering (DVCS) 
    """

    _PARAMETER_A = 0.50
    _PARAMETER_B = -0.41
    _PARAMETER_C = 0.05
    _PARAMETER_D = -0.25
    _PARAMETER_E = 0.55
    _PARAMETER_F = 0.166

    return generic_compton_form_factor_function(
        x_Bjorken,
        t_squared_hadronic_momentum_transfer,
        _PARAMETER_A,
        _PARAMETER_B,
        _PARAMETER_C,
        _PARAMETER_D,
        _PARAMETER_E,
        _PARAMETER_F)