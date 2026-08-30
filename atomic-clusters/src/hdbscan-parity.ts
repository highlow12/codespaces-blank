import type { HdbscanOutput } from "./clustering";

export interface HdbscanParityMetrics {
  referenceClusters: number;
  candidateClusters: number;
  labelAgreement: number;
  adjustedRandIndex: number;
  noiseAgreement: number;
  noiseExact: boolean;
  probabilityMae: number;
  probabilityRmse: number;
  probabilityMaxError: number;
  outlierMae: number;
  outlierMaxError: number;
  membershipMae: number;
  membershipMaxError: number;
  mapping: Record<string, number>;
}

/**
 * Compare two HDBSCAN results while ignoring arbitrary cluster-number
 * permutations.  This deliberately reports probability and outlier error
 * separately: equal labels alone are not membership parity.
 */
export function compareHdbscanOutputs(reference: HdbscanOutput, candidate: HdbscanOutput): HdbscanParityMetrics {
  const rowCount = reference.labels.length;
  if (candidate.labels.length !== rowCount || candidate.probabilities.length !== rowCount) throw new TypeError("HDBSCAN parity results must have equal row counts");
  const refLabels = integerLabels(reference.labels, rowCount, "reference");
  const candidateLabels = integerLabels(candidate.labels, rowCount, "candidate");
  const referenceProbabilities = scoreVector(reference.probabilities, rowCount, "reference probabilities");
  const candidateProbabilities = scoreVector(candidate.probabilities, rowCount, "candidate probabilities");
  const refClusters = clusterCount(refLabels); const candidateClusters = clusterCount(candidateLabels);
  const mapping = bestClusterMapping(refLabels, candidateLabels, refClusters, candidateClusters);
  const alignedLabels = candidateLabels.map((label) => label < 0 ? -1 : (mapping.get(label) ?? -1));
  const labelAgreement = mean(alignedLabels.map((label, index) => label === refLabels[index] ? 1 : 0));
  const noiseAgreement = mean(alignedLabels.map((label, index) => (label < 0) === (refLabels[index] < 0) ? 1 : 0));
  const referenceOutliers = scoreVector(reference.outlierProxy || referenceProbabilities.map((value) => 1 - value), rowCount, "reference outliers");
  const candidateOutliers = scoreVector(candidate.outlierProxy || candidateProbabilities.map((value) => 1 - value), rowCount, "candidate outliers");
  const probabilityErrors = referenceProbabilities.map((value, index) => value - candidateProbabilities[index]);
  const outlierErrors = referenceOutliers.map((value, index) => value - candidateOutliers[index]);
  const referenceMemberships = membershipMatrix(reference, rowCount, refClusters);
  const candidateMemberships = membershipMatrix(candidate, rowCount, candidateClusters);
  const membershipErrors: number[] = [];
  for (let row = 0; row < rowCount; row++) for (let cluster = 0; cluster < refClusters; cluster++) {
    let value = 0;
    for (let candidateCluster = 0; candidateCluster < candidateClusters; candidateCluster++) if (mapping.get(candidateCluster) === cluster) value += candidateMemberships[row][candidateCluster];
    membershipErrors.push(referenceMemberships[row][cluster] - value);
  }
  return {
    referenceClusters: refClusters, candidateClusters, labelAgreement,
    adjustedRandIndex: adjustedRandIndex(refLabels, candidateLabels), noiseAgreement, noiseExact: noiseAgreement === 1,
    probabilityMae: meanAbsolute(probabilityErrors), probabilityRmse: rootMeanSquare(probabilityErrors),
    probabilityMaxError: maximumAbsolute(probabilityErrors), outlierMae: meanAbsolute(outlierErrors), outlierMaxError: maximumAbsolute(outlierErrors), membershipMae: meanAbsolute(membershipErrors),
    membershipMaxError: membershipErrors.reduce((maximum, value) => Math.max(maximum, Math.abs(value)), 0),
    mapping: Object.fromEntries([...mapping.entries()].sort((a, b) => a[0] - b[0]).map(([key, value]) => [String(key), value]))
  };
}

/** Adjusted Rand index over the complete partition, including noise as a category. */
function adjustedRandIndex(reference: number[], candidate: number[]): number {
  if (reference.length !== candidate.length) throw new TypeError("ARI label vectors must have equal row counts");
  if (reference.length < 2) return 1;
  const refCounts = new Map<number, number>(); const candidateCounts = new Map<number, number>(); const cells = new Map<string, number>();
  for (let index = 0; index < reference.length; index++) {
    const ref = reference[index]; const item = candidate[index];
    refCounts.set(ref, (refCounts.get(ref) || 0) + 1); candidateCounts.set(item, (candidateCounts.get(item) || 0) + 1);
    const key = `${ref}:${item}`; cells.set(key, (cells.get(key) || 0) + 1);
  }
  const pairs = (count: number): number => count * (count - 1) / 2;
  const cellPairs = [...cells.values()].reduce((sum, count) => sum + pairs(count), 0);
  const refPairs = [...refCounts.values()].reduce((sum, count) => sum + pairs(count), 0);
  const candidatePairs = [...candidateCounts.values()].reduce((sum, count) => sum + pairs(count), 0);
  const totalPairs = pairs(reference.length); const expected = refPairs * candidatePairs / totalPairs; const maximum = (refPairs + candidatePairs) / 2; const denominator = maximum - expected;
  if (Math.abs(denominator) <= Number.EPSILON) return cellPairs === maximum ? 1 : 0;
  return (cellPairs - expected) / denominator;
}

function integerLabels(values: number[], rowCount: number, source: string): number[] {
  if (values.length !== rowCount || values.some((value) => !Number.isSafeInteger(value) || value < -1)) throw new TypeError(`${source} labels are invalid`);
  return values;
}
function scoreVector(values: number[], rowCount: number, source: string): number[] {
  if (!Array.isArray(values) || values.length !== rowCount || values.some((value) => !Number.isFinite(value) || value < 0 || value > 1)) throw new TypeError(`${source} must be a finite vector in [0, 1]`);
  return values;
}
function clusterCount(labels: number[]): number { return labels.reduce((maximum, label) => Math.max(maximum, label + 1), 0); }
function membershipMatrix(output: HdbscanOutput, rows: number, clusters: number): number[][] {
  if (output.memberships !== undefined) {
    if (output.memberships.length !== rows || output.memberships.some((row) => row.length !== clusters)) throw new TypeError("HDBSCAN membership matrices must match their label cluster count");
    if (output.memberships.some((row) => row.some((value) => !Number.isFinite(value) || value < 0 || value > 1) || row.reduce((sum, value) => sum + value, 0) > 1 + 1e-6)) throw new TypeError("HDBSCAN memberships must be finite probabilities with row sums at most one");
    return output.memberships;
  }
  return output.labels.map((label, row) => Array.from({ length: clusters }, (_, cluster) => label === cluster ? output.probabilities[row] : 0));
}
function bestClusterMapping(reference: number[], candidate: number[], refClusters: number, candidateClusters: number): Map<number, number> {
  const mapping = new Map<number, number>();
  if (!candidateClusters || !refClusters) return mapping;
  const overlap = Array.from({ length: candidateClusters }, () => new Array(refClusters).fill(0));
  for (let row = 0; row < reference.length; row++) if (candidate[row] >= 0 && reference[row] >= 0) overlap[candidate[row]][reference[row]]++;
  // Add one zero-weight dummy column per candidate. This makes the matrix
  // rectangular-safe and lets a surplus candidate cluster remain unmapped.
  const columns = Math.max(refClusters, candidateClusters);
  const costs = Array.from({ length: candidateClusters }, (_, candidateCluster) => Array.from({ length: columns }, (_, ref) => ref < refClusters ? -overlap[candidateCluster][ref] : 0));
  const assignment = minimumCostAssignment(costs);
  assignment.forEach((ref, candidateCluster) => { if (ref >= 0 && ref < refClusters) mapping.set(candidateCluster, ref); });
  return mapping;
}

/** Deterministic Hungarian assignment for a cost matrix with rows <= columns. */
function minimumCostAssignment(costs: number[][]): number[] {
  const rows = costs.length; const columns = costs[0]?.length || 0;
  if (!rows || rows > columns) throw new TypeError("assignment matrix must have no more rows than columns");
  const u = new Array(rows + 1).fill(0); const v = new Array(columns + 1).fill(0);
  const columnOwner = new Array(columns + 1).fill(0); const previousColumn = new Array(columns + 1).fill(0);
  for (let row = 1; row <= rows; row++) {
    columnOwner[0] = row; let activeColumn = 0;
    const minimum = new Array(columns + 1).fill(Infinity); const used = new Array(columns + 1).fill(false);
    do {
      used[activeColumn] = true; const activeRow = columnOwner[activeColumn]; let delta = Infinity; let nextColumn = 0;
      for (let column = 1; column <= columns; column++) if (!used[column]) {
        const reduced = costs[activeRow - 1][column - 1] - u[activeRow] - v[column];
        if (reduced < minimum[column]) { minimum[column] = reduced; previousColumn[column] = activeColumn; }
        if (minimum[column] < delta) { delta = minimum[column]; nextColumn = column; }
      }
      for (let column = 0; column <= columns; column++) if (used[column]) { u[columnOwner[column]] += delta; v[column] -= delta; } else minimum[column] -= delta;
      activeColumn = nextColumn;
    } while (columnOwner[activeColumn] !== 0);
    do {
      const prior = previousColumn[activeColumn]; columnOwner[activeColumn] = columnOwner[prior]; activeColumn = prior;
    } while (activeColumn !== 0);
  }
  const assignment = new Array(rows).fill(-1);
  for (let column = 1; column <= columns; column++) if (columnOwner[column] > 0) assignment[columnOwner[column] - 1] = column - 1;
  return assignment;
}
function mean(values: number[]): number { return values.length ? values.reduce((sum, value) => sum + value, 0) / values.length : 1; }
function meanAbsolute(values: number[]): number { return values.length ? values.reduce((sum, value) => sum + Math.abs(value), 0) / values.length : 0; }
function maximumAbsolute(values: number[]): number { return values.reduce((maximum, value) => Math.max(maximum, Math.abs(value)), 0); }
function rootMeanSquare(values: number[]): number { return values.length ? Math.sqrt(values.reduce((sum, value) => sum + value * value, 0) / values.length) : 0; }
