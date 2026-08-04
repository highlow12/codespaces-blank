"""Compatibility facade for the split FCM implementation modules."""

from fcm_core import (
    DEFAULT_CLUSTERING_PCA_COMPONENTS,
    conditional_memberships_from_projected,
    fcm_memberships_from_centers,
    fit_clustering_pca,
    fit_pca_normalized_features,
    pca_normalized_features,
    spherical_fcm,
    transform_pca_normalized_features,
)
from fcm_document_classification import (
    DEFAULT_FORCED_NOISE_RATIO,
    DEFAULT_MAX_MEMBERSHIP_GAP,
    classify_fcm_documents,
    fcm_document_types,
    fcm_membership_boundary_mask,
    fcm_noise_mask,
    fcm_noise_scores,
    forced_noise_mask,
    merge_forced_noise,
)
from fcm_validity import (
    FCM_SELECTION_METHODS,
    _filter_fcm_labels,
    fuzzy_silhouette_proxy,
    modified_partition_coefficient,
    normalized_partition_entropy,
    partition_coefficient,
    partition_entropy,
    select_fcm_cluster_count,
    spherical_fcm_objective,
    xie_beni_index,
)
from hierarchical_assignments import (
    DOCUMENT_TYPE_BOUNDARY,
    DOCUMENT_TYPE_CORE,
    DOCUMENT_TYPE_NOISE,
    path_membership_column,
)
from hierarchical_fcm import run_hierarchical_pca_fcm


__all__ = [
    "DEFAULT_CLUSTERING_PCA_COMPONENTS",
    "DEFAULT_FORCED_NOISE_RATIO",
    "DEFAULT_MAX_MEMBERSHIP_GAP",
    "DOCUMENT_TYPE_BOUNDARY",
    "DOCUMENT_TYPE_CORE",
    "DOCUMENT_TYPE_NOISE",
    "FCM_SELECTION_METHODS",
    "classify_fcm_documents",
    "conditional_memberships_from_projected",
    "fcm_document_types",
    "fcm_membership_boundary_mask",
    "fcm_memberships_from_centers",
    "fcm_noise_mask",
    "fcm_noise_scores",
    "fit_pca_normalized_features",
    "fit_clustering_pca",
    "forced_noise_mask",
    "fuzzy_silhouette_proxy",
    "merge_forced_noise",
    "modified_partition_coefficient",
    "normalized_partition_entropy",
    "partition_coefficient",
    "partition_entropy",
    "path_membership_column",
    "pca_normalized_features",
    "run_hierarchical_pca_fcm",
    "select_fcm_cluster_count",
    "spherical_fcm",
    "spherical_fcm_objective",
    "transform_pca_normalized_features",
    "xie_beni_index",
]
