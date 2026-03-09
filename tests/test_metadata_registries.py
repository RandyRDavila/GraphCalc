import importlib

from graphcalc.metadata import build_module_registry


EXPECTED_REGISTRIES = {
    "graphcalc.hypergraphs.invariants.acyclicity": {
        "is_alpha_acyclic": "Alpha-acyclicity",
        "berge_girth": "Berge girth",
        "is_berge_acyclic": "Berge-acyclicity",
    },
    "graphcalc.hypergraphs.invariants.basic": {
        "number_of_vertices": "Number of vertices",
        "number_of_edges": "Number of edges",
        "is_empty": "Emptiness",
        "rank": "Rank",
        "co_rank": "Co-rank",
        "maximum_degree": "Maximum degree",
        "minimum_degree": "Minimum degree",
        "average_degree": "Average degree",
    },
    "graphcalc.hypergraphs.invariants.chromatic": {
        "weak_coloring": "Weak coloring",
        "weak_chromatic_number": "Weak chromatic number",
        "strong_coloring": "Strong coloring",
        "strong_chromatic_number": "Strong chromatic number",
        "edge_coloring": "Edge coloring",
        "edge_chromatic_number": "Edge chromatic number",
    },
    "graphcalc.hypergraphs.invariants.codegree": {
        "codegree": "Codegree",
        "maximum_codegree": "Maximum t-codegree",
        "minimum_codegree": "Minimum t-codegree",
        "average_codegree": "Average t-codegree",
        "lower_shadow": "Lower shadow",
        "lower_shadow_size": "Lower shadow size",
        "upper_shadow": "Upper shadow",
        "upper_shadow_size": "Upper shadow size",
    },
    "graphcalc.hypergraphs.invariants.configurations": {
        "has_sunflower": "Sunflower existence",
    },
    "graphcalc.hypergraphs.invariants.domination": {
        "minimum_dominating_set": "Minimum dominating set",
        "domination_number": "Domination number",
        "minimum_total_dominating_set": "Minimum total dominating set",
        "total_domination_number": "Total domination number",
    },
    "graphcalc.hypergraphs.invariants.dsi": {
        "degree_sequence": "Degree sequence",
        "reverse_degree_sequence": "Reverse degree sequence",
        "hh_residue_graph_degree_sequence": "Havel--Hakimi residue",
        "generalized_havel_hakimi_residue": "Generalized Havel--Hakimi residue",
        "generalized_annihilation_number": "Generalized annihilation number",
    },
    "graphcalc.hypergraphs.invariants.independence": {
        "maximum_independent_set": "Maximum independent set",
        "independence_number": "Independence number",
    },
    "graphcalc.hypergraphs.invariants.matching": {
        "maximum_matching": "Maximum matching",
        "matching_number": "Matching number",
        "fractional_matching_number": "Fractional matching number",
        "minimum_edge_cover": "Minimum edge cover",
        "edge_cover_number": "Edge cover number",
    },
    "graphcalc.hypergraphs.invariants.partite": {
        "is_r_partite_r_uniform": "r-partite r-uniform property",
    },
    "graphcalc.hypergraphs.invariants.structure": {
        "is_simple": "Simplicity",
        "is_linear": "Linearity",
        "is_intersecting": "Intersecting property",
        "is_pair_covering": "Pair-covering property",
        "is_sperner": "Sperner property",
        "is_clutter": "Clutter property",
        "is_t_intersecting": "t-intersecting property",
    },
    "graphcalc.hypergraphs.invariants.transversals": {
        "minimum_transversal": "Minimum transversal",
        "transversal_number": "Transversal number",
        "fractional_transversal_number": "Fractional transversal number",
    },
    "graphcalc.quantum.invariants": {
        "purity": "Purity",
        "rank": "Rank",
        "linear_entropy": "Linear entropy",
        "von_neumann_entropy": "von Neumann entropy",
        "negativity": "Negativity",
        "logarithmic_negativity": "Logarithmic negativity",
        "fidelity": "Fidelity",
        "entanglement_entropy": "Entanglement entropy",
        "mutual_information": "Mutual information",
    },
    "graphcalc.quantum.properties": {
        "is_valid_state": "Valid state property",
        "is_pure": "Purity property",
        "is_mixed": "Mixed-state property",
        "has_positive_partial_transpose": "Positive partial transpose property",
        "is_product_state": "Product-state property",
        "is_entangled": "Entanglement property",
    },
    "graphcalc.quantum.channel_invariants": {
        "choi_rank": "Choi rank",
        "kraus_rank": "Kraus rank",
        "input_dimension": "Input dimension",
        "output_dimension": "Output dimension",
        "choi_eigenvalues": "Choi eigenvalues",
    },
    "graphcalc.quantum.channel_properties": {
        "is_completely_positive": "Complete positivity",
        "is_trace_preserving": "Trace-preserving property",
        "is_unital": "Unital property",
        "is_quantum_channel": "Quantum channel property",
        "is_unitary_channel": "Unitary channel property",
    },
    "graphcalc.quantum.measurement_properties": {
        "is_povm": "POVM property",
        "is_projective_measurement": "Projective measurement property",
        "is_rank_one_measurement": "Rank-one measurement property",
    },
}


def test_expected_registries_present():
    for module_name, expected in EXPECTED_REGISTRIES.items():
        mod = importlib.import_module(module_name)
        registry = build_module_registry(mod)

        for key, display_name in expected.items():
            assert key in registry
            assert registry[key]["display_name"] == display_name
            assert registry[key]["definition"]
            assert registry[key]["category"]
