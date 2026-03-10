from .basic import *
from .structure import *
from .codegree import *
from .transversals import *
from .independence import *
from .matching import *
from .acyclicity import *
from .partite import *
from .configurations import *
from .domination import *
from .dsi import *

__all__ = [
    "number_of_vertices",
    "number_of_edges",
    "is_empty",
    "is_trivial",
    "rank",
    "co_rank",
    "is_k_uniform",
    "maximum_degree",
    "minimum_degree",
    "average_degree",
    "degree_sequence",
    "edge_size_sequence",
    "is_regular",
    "is_d_regular",
    "is_simple",
    "is_linear",
    "is_intersecting",
    "is_pair_covering",
    "is_sperner",
    "codegree",
    "maximum_codegree",
    "minimum_codegree",
    "average_codegree",
    "lower_shadow",
    "lower_shadow_size",
    "upper_shadow",
    "upper_shadow_size",
    "minimum_transversal",
    "transversal_number",
    "fractional_transversal_number",
    "maximum_independent_set",
    "independence_number",
    "maximum_matching",
    "matching_number",
    "fractional_matching_number",
    "minimum_edge_cover",
    "edge_cover_number",
    "is_clutter",
    "is_t_intersecting",
    "is_alpha_acyclic",
    "berge_girth",
    "is_berge_acyclic",
    "is_r_partite_r_uniform",
    "has_sunflower",
    "reverse_degree_sequence",
    "generalized_havel_hakimi_residue",
    "generalized_annihilation_number",
    "minimum_dominating_set",
    "domination_number",
    "minimum_total_dominating_set",
    "total_domination_number",
]
