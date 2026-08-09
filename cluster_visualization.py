"""Compatibility facade for projection, assignment, and plotting modules."""

from cluster_plotting import (
    NOISE_COLOR,
    categorical_color_map,
    compact_umap_presets,
    hierarchical_color_map,
    make_cluster_plot,
    make_comparison_plot,
    make_fixed_coordinate_plot,
    make_selected_coordinate_plot,
)
from visualization_constants import (
    DEFAULT_CLUSTER_TARGET_WEIGHT,
    DEFAULT_VISUAL_PCA_COMPONENTS,
)
from umap_projection import (
    _load_umap,
    _make_umap_reducer,
    _validate_cluster_target,
    fit_projection_model,
    load_embeddings,
    project_embeddings,
    transform_projection,
)
from visual_assignments import (
    build_cluster_supervision,
    load_assignments,
    prepare_visual_assignments,
)


__all__ = [
    "DEFAULT_CLUSTER_TARGET_WEIGHT",
    "DEFAULT_VISUAL_PCA_COMPONENTS",
    "NOISE_COLOR",
    "build_cluster_supervision",
    "categorical_color_map",
    "compact_umap_presets",
    "fit_projection_model",
    "hierarchical_color_map",
    "load_assignments",
    "load_embeddings",
    "make_cluster_plot",
    "make_comparison_plot",
    "make_fixed_coordinate_plot",
    "make_selected_coordinate_plot",
    "prepare_visual_assignments",
    "project_embeddings",
    "transform_projection",
]
