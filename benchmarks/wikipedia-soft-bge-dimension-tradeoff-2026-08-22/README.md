# Wikipedia BGE projection dimension trade-off

This benchmark compares centered PCA with uncentered TruncatedSVD on the 720-document BGE embedding set. Every projection is fit on the 432 discovery documents only. Calibration (144 documents) is used for the complete 3x3x3x3 HDBSCAN/membership sweep and configuration selection; the 144 test documents are evaluated only after that selection.

The common curve uses 32, 64, 128, and 256 dimensions. Centered PCA on 432 rows has maximum centered rank 431, so a 512-dimensional centered point is invalid (and cannot be compared fairly with the uncentered method).

tradeoff_curve.csv reports, for calibration and test, Pearson/Spearman correlation to original BGE cross-cosines, cosine and projected-Euclidean kNN recall@24, leaf neighbour purity, and same-leaf minus different-leaf cosine margin. It also reports native and exact-kNN test cluster quality using the calibration-selected sweep setting. tradeoff_curve.png plots the test curves and exact-kNN leaf NMI.
