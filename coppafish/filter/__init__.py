try:
    from .deconvolution_pytorch import wiener_deconvolve
except ImportError:
    from .deconvolution import wiener_deconvolve
