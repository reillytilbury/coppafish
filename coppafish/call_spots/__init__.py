try:
    from .qual_check_optimised import get_spot_intensity
except ImportError:
    try:
        from .qual_check_pytorch import get_spot_intensity
    except ImportError:
        from .qual_check import get_spot_intensity
try:
    import jax
    from .background import fit_background
except ImportError:
    try:
        from .background_pytorch import fit_background
    except ImportError:
        from .background import fit_background

from .base import get_bled_codes, get_non_duplicate, compute_gene_efficiency
from .qual_check import omp_spot_score, quality_threshold, get_intensity_thresh
from .dot_product import dot_product_score, gene_prob_score
# Needed for working non-jax, non-pytorch software
from . import background
