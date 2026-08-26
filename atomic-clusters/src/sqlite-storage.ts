/**
 * Durable, self contained storage for Atomic Clusters.
 *
 * The module intentionally talks to the small part of the sql.js API used by
 * the plugin.  This keeps the database usable in Node tests and makes the
 * sql.js initialiser (and its wasm asset path) an explicit application concern.
 */
import { contentHash } from "./hash";
import { CachedEmbedding, ClusterResult, EmbeddingRunLog, NoteRecord } from "./types";

export interface SqlValue { readonly [key: string]: unknown; }
export interface SqlStatement {
  bind(values?: unknown[] | Record<string, unknown>): void;
  step(): boolean;
  getAsObject(): Record<string, unknown>;
  free(): void;
}
export interface SqlDatabase {
  run(sql: string, params?: unknown[] | Record<string, unknown>): void;
  prepare(sql: string): SqlStatement;
  exec(sql: string): Array<{ columns: string[]; values: unknown[][] }>;
  export(): Uint8Array;
  close(): void;
}
export interface SqlJsStatic { Database: new (data?: ArrayLike<number>) => SqlDatabase; }

export interface BinaryAdapter {
  readBinary(path: string): Promise<ArrayBuffer>;
  writeBinary(path: string, data: ArrayBuffer): Promise<void>;
  exists(path: string): Promise<boolean>;
  mkdir(path: string): Promise<void>;
  rename?(oldPath: string, newPath: string): Promise<void>;
  remove?(path: string): Promise<void>;
}

export const SQLITE_PATH = ".obsidian/plugins/atomic-clusters/atomic-clusters.sqlite";
export const SQLITE_TEMP_PATH = `${SQLITE_PATH}.tmp`;
export const SQLITE_SCHEMA_VERSION = 5;

const SCHEMA = `
PRAGMA foreign_keys = ON;
CREATE TABLE IF NOT EXISTS metadata (key TEXT PRIMARY KEY, value TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS notes (
  path TEXT PRIMARY KEY, title TEXT NOT NULL, mtime INTEGER NOT NULL,
  content_hash TEXT NOT NULL, content TEXT
);
CREATE TABLE IF NOT EXISTS embeddings (
  path TEXT NOT NULL, provider TEXT NOT NULL, model TEXT NOT NULL,
  embedding_hash TEXT NOT NULL, note_content_hash TEXT NOT NULL, dimension INTEGER NOT NULL, vector_json TEXT NOT NULL,
  created_at TEXT NOT NULL, PRIMARY KEY(path, provider, model, embedding_hash),
  FOREIGN KEY(path) REFERENCES notes(path) ON DELETE CASCADE
);
CREATE INDEX IF NOT EXISTS embeddings_current ON embeddings(path, provider, model, created_at DESC);
CREATE TABLE IF NOT EXISTS pca_models (
  model_hash TEXT PRIMARY KEY, provider TEXT, model TEXT, input_dimension INTEGER NOT NULL,
  output_dimension INTEGER NOT NULL, normalization TEXT NOT NULL, mean_json TEXT NOT NULL,
  components_json TEXT NOT NULL, explained_variance_json TEXT NOT NULL,
  created_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS pca_coordinates (
  path TEXT NOT NULL, model_hash TEXT NOT NULL, coordinates_json TEXT NOT NULL,
  PRIMARY KEY(path, model_hash), FOREIGN KEY(path) REFERENCES notes(path) ON DELETE CASCADE,
  FOREIGN KEY(model_hash) REFERENCES pca_models(model_hash) ON DELETE CASCADE
);
CREATE TABLE IF NOT EXISTS results (
  result_id TEXT PRIMARY KEY, schema_version INTEGER NOT NULL, created_at TEXT NOT NULL,
  result_json TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS assignments (
  result_id TEXT NOT NULL, path TEXT NOT NULL, leaf_label INTEGER NOT NULL,
  probability REAL NOT NULL, outlier_score REAL NOT NULL, PRIMARY KEY(result_id, path),
  FOREIGN KEY(result_id) REFERENCES results(result_id) ON DELETE CASCADE,
  FOREIGN KEY(path) REFERENCES notes(path) ON DELETE CASCADE
);
CREATE TABLE IF NOT EXISTS hierarchy_merges (
  result_id TEXT NOT NULL, id INTEGER NOT NULL, left_id INTEGER NOT NULL,
  right_id INTEGER NOT NULL, distance REAL NOT NULL, mass INTEGER NOT NULL,
  PRIMARY KEY(result_id, id), FOREIGN KEY(result_id) REFERENCES results(result_id) ON DELETE CASCADE
);
CREATE TABLE IF NOT EXISTS hierarchy_leaves (
  result_id TEXT NOT NULL, ordinal INTEGER NOT NULL, leaf_id INTEGER NOT NULL,
  PRIMARY KEY(result_id, ordinal), FOREIGN KEY(result_id) REFERENCES results(result_id) ON DELETE CASCADE
);
CREATE TABLE IF NOT EXISTS leaf_order (
  result_id TEXT NOT NULL, ordinal INTEGER NOT NULL, leaf_id INTEGER NOT NULL,
  PRIMARY KEY(result_id, ordinal), FOREIGN KEY(result_id) REFERENCES results(result_id) ON DELETE CASCADE
);
CREATE TABLE IF NOT EXISTS visualization_points (
  result_id TEXT NOT NULL, ordinal INTEGER NOT NULL, path TEXT NOT NULL,
  x REAL NOT NULL, y REAL NOT NULL, leaf_label INTEGER NOT NULL,
  PRIMARY KEY(result_id, ordinal), FOREIGN KEY(result_id) REFERENCES results(result_id) ON DELETE CASCADE,
  FOREIGN KEY(path) REFERENCES notes(path) ON DELETE CASCADE
);
CREATE TABLE IF NOT EXISTS cluster_titles (
  result_id TEXT NOT NULL, node_id INTEGER NOT NULL, title TEXT NOT NULL,
  PRIMARY KEY(result_id, node_id), FOREIGN KEY(result_id) REFERENCES results(result_id) ON DELETE CASCADE
);
CREATE TABLE IF NOT EXISTS soft_memberships (
  result_id TEXT NOT NULL, path TEXT NOT NULL, leaf_id INTEGER NOT NULL, membership REAL NOT NULL,
  PRIMARY KEY(result_id, path, leaf_id), FOREIGN KEY(result_id) REFERENCES results(result_id) ON DELETE CASCADE,
  FOREIGN KEY(path) REFERENCES notes(path) ON DELETE CASCADE
);
CREATE TABLE IF NOT EXISTS embedding_logs (
  id INTEGER PRIMARY KEY AUTOINCREMENT, started_at TEXT NOT NULL, completed_at TEXT NOT NULL,
  provider TEXT NOT NULL, model TEXT NOT NULL, status TEXT, log_json TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS migrations (
  name TEXT PRIMARY KEY, completed_at TEXT NOT NULL
);
CREATE VIEW IF NOT EXISTS v_current_embeddings AS
  SELECT e.path, n.title, n.mtime, n.content_hash, e.provider, e.model,
    e.embedding_hash, e.note_content_hash, e.dimension, e.vector_json, e.created_at
  FROM embeddings e JOIN notes n USING(path)
  WHERE e.created_at = (SELECT MAX(e2.created_at) FROM embeddings e2
                        WHERE e2.path=e.path AND e2.provider=e.provider AND e2.model=e.model);
CREATE VIEW IF NOT EXISTS v_note_pca AS
  SELECT c.path, n.title, n.mtime, n.content_hash, c.model_hash, c.coordinates_json
  FROM pca_coordinates c JOIN notes n USING(path);
CREATE VIEW IF NOT EXISTS v_cluster_assignments AS
  SELECT a.result_id, a.path, n.title, a.leaf_label, a.probability, a.outlier_score
  FROM assignments a JOIN notes n USING(path);
CREATE VIEW IF NOT EXISTS v_embedding_log AS
  SELECT id, started_at, completed_at, provider, model, status, log_json FROM embedding_logs;
`;

export interface PcaModel {
  modelHash: string;
  inputDimension: number;
  outputDimension: number;
  /** One mean value per input dimension. */
  mean: number[];
  /** Components are rows: outputDimension × inputDimension. */
  components: number[][];
  explainedVariance: number[];
  normalization: "l2" | "none" | string;
  provider?: string;
  model?: string;
}

/** Apply the exact projection used by the browser and Python implementations. */
export function projectPca(vector: number[], pca: PcaModel): number[] {
  if (vector.length !== pca.inputDimension || pca.mean.length !== pca.inputDimension) {
    throw new Error(`PCA input dimension mismatch: expected ${pca.inputDimension}, got ${vector.length}`);
  }
  const normalized = vector.slice();
  if (pca.normalization === "l2") {
    const norm = Math.sqrt(normalized.reduce((sum, value) => sum + value * value, 0));
    if (norm > 0) for (let i = 0; i < normalized.length; i++) normalized[i] /= norm;
  }
  const centered = normalized.map((value, index) => value - pca.mean[index]);
  return pca.components.map((component) => {
    if (component.length !== pca.inputDimension) throw new Error("PCA component dimension mismatch");
    return component.reduce((sum, value, index) => sum + value * centered[index], 0);
  });
}

/** Stable vector fingerprint; includes dimensionality and IEEE-754 values. */
export async function embeddingHash(vector: number[]): Promise<string> {
  return contentHash(JSON.stringify({ dimension: vector.length, vector }));
}

export async function pcaModelHash(model: Omit<PcaModel, "modelHash">): Promise<string> {
  return contentHash(JSON.stringify({ inputDimension: model.inputDimension, outputDimension: model.outputDimension,
    normalization: model.normalization, mean: model.mean, components: model.components,
    explainedVariance: model.explainedVariance, provider: model.provider || "", model: model.model || "" }));
}

export interface SqliteStorageOptions { path?: string; now?: () => string; }

export class SqliteClusterStore {
  readonly path: string;
  private db!: SqlDatabase;
  private opened = false;
  private readonly now: () => string;
  constructor(private readonly adapter: BinaryAdapter, private readonly sql: SqlJsStatic, options: SqliteStorageOptions = {}) {
    this.path = options.path || SQLITE_PATH;
    this.now = options.now || (() => new Date().toISOString());
  }

  async open(): Promise<this> {
    let bytes: ArrayBuffer | undefined;
    if (await this.adapter.exists(this.path)) bytes = await this.adapter.readBinary(this.path);
    this.db = new this.sql.Database(bytes ? new Uint8Array(bytes) : undefined);
    this.db.run(SCHEMA);
    this.db.run("INSERT OR REPLACE INTO metadata(key,value) VALUES('schema_version',?)", [String(SQLITE_SCHEMA_VERSION)]);
    this.opened = true;
    return this;
  }
  get database(): SqlDatabase { this.requireOpen(); return this.db; }
  close(): void { if (this.opened) { this.db.close(); this.opened = false; } }
  private requireOpen(): void { if (!this.opened) throw new Error("SQLite store is not open"); }

  async flush(): Promise<void> {
    this.requireOpen();
    const parent = this.path.slice(0, this.path.lastIndexOf("/"));
    if (parent && !(await this.adapter.exists(parent))) await this.adapter.mkdir(parent);
    const bytes = this.db.export();
    const durableBytes = bytes.slice().buffer as ArrayBuffer;
    await this.adapter.writeBinary(`${this.path}.tmp`, durableBytes);
    if (this.adapter.rename) await this.adapter.rename(`${this.path}.tmp`, this.path);
    else {
      await this.adapter.writeBinary(this.path, durableBytes);
      await this.adapter.remove?.(`${this.path}.tmp`);
    }
  }

  /** Execute a mutating operation and persist it as one transaction. */
  async transaction<T>(operation: (db: SqlDatabase) => T | Promise<T>): Promise<T> {
    this.requireOpen();
    this.db.run("BEGIN IMMEDIATE");
    try { const result = await operation(this.db); this.db.run("COMMIT"); await this.flush(); return result; }
    catch (error) { try { this.db.run("ROLLBACK"); } catch { /* preserve original error */ } throw error; }
  }

  async upsertNote(note: NoteRecord): Promise<void> {
    await this.transaction((db) => db.run("INSERT INTO notes(path,title,mtime,content_hash,content) VALUES(?,?,?,?,?) ON CONFLICT(path) DO UPDATE SET title=excluded.title,mtime=excluded.mtime,content_hash=excluded.content_hash,content=excluded.content", [note.path, note.title, note.mtime, note.hash, note.content]));
  }
  async upsertNotes(notes: NoteRecord[]): Promise<void> {
    await this.transaction((db) => { for (const note of notes) db.run("INSERT INTO notes(path,title,mtime,content_hash,content) VALUES(?,?,?,?,?) ON CONFLICT(path) DO UPDATE SET title=excluded.title,mtime=excluded.mtime,content_hash=excluded.content_hash,content=excluded.content", [note.path, note.title, note.mtime, note.hash, note.content]); });
  }
  async putEmbedding(entry: CachedEmbedding): Promise<string> {
    const hash = await embeddingHash(entry.vector);
    await this.transaction((db) => db.run("INSERT OR IGNORE INTO embeddings(path,provider,model,embedding_hash,note_content_hash,dimension,vector_json,created_at) VALUES(?,?,?,?,?,?,?,?)", [entry.path, entry.provider, entry.model, hash, entry.hash, entry.vector.length, JSON.stringify(entry.vector), this.now()]));
    return hash;
  }
  async getEmbedding(path: string, provider: string, model: string, noteHash?: string): Promise<CachedEmbedding | undefined> {
    this.requireOpen();
    const statement = this.db.prepare("SELECT e.path,e.provider,e.model,e.embedding_hash,e.note_content_hash,e.vector_json,n.content_hash FROM embeddings e JOIN notes n USING(path) WHERE e.path=? AND e.provider=? AND e.model=? ORDER BY e.created_at DESC");
    try { statement.bind([path, provider, model]); while (statement.step()) { const row = statement.getAsObject(); if (!noteHash || row.note_content_hash === noteHash || row.content_hash === noteHash) return { path: String(row.path), provider: String(row.provider), model: String(row.model), hash: String(row.note_content_hash), vector: JSON.parse(String(row.vector_json)) }; } }
    finally { statement.free(); }
    return undefined;
  }
  async savePcaModel(model: PcaModel): Promise<void> {
    await this.transaction((db) => db.run("INSERT OR REPLACE INTO pca_models(model_hash,provider,model,input_dimension,output_dimension,normalization,mean_json,components_json,explained_variance_json,created_at) VALUES(?,?,?,?,?,?,?,?,?,?)", [model.modelHash, model.provider || null, model.model || null, model.inputDimension, model.outputDimension, model.normalization, JSON.stringify(model.mean), JSON.stringify(model.components), JSON.stringify(model.explainedVariance), this.now()]));
  }
  async getPcaModel(modelHash: string): Promise<PcaModel | undefined> {
    this.requireOpen(); const rows = this.db.exec("SELECT * FROM pca_models WHERE model_hash=" + sqlQuote(modelHash));
    const row = rows[0]?.values[0]; if (!row) return undefined;
    const columns = rows[0].columns; const object = Object.fromEntries(columns.map((key, i) => [key, row[i]]));
    return { modelHash: String(object.model_hash), provider: object.provider == null ? undefined : String(object.provider), model: object.model == null ? undefined : String(object.model), inputDimension: Number(object.input_dimension), outputDimension: Number(object.output_dimension), normalization: String(object.normalization), mean: JSON.parse(String(object.mean_json)), components: JSON.parse(String(object.components_json)), explainedVariance: JSON.parse(String(object.explained_variance_json)) };
  }
  async project(path: string, vector: number[], model: PcaModel): Promise<number[]> {
    const coordinates = projectPca(vector, model);
    await this.transaction((db) => db.run("INSERT OR REPLACE INTO pca_coordinates(path,model_hash,coordinates_json) VALUES(?,?,?)", [path, model.modelHash, JSON.stringify(coordinates)]));
    return coordinates;
  }
  async saveResult(result: ClusterResult, options: { resultId?: string; coordinates?: Record<string, number[]>; softMemberships?: Record<string, Record<number, number>> } = {}): Promise<string> {
    const resultId = options.resultId || await contentHash(JSON.stringify(result));
    await this.transaction((db) => {
      db.run("INSERT OR REPLACE INTO results(result_id,schema_version,created_at,result_json) VALUES(?,?,?,?)", [resultId, result.schemaVersion, this.now(), JSON.stringify(result)]);
      db.run("DELETE FROM assignments WHERE result_id=?", [resultId]); db.run("DELETE FROM hierarchy_merges WHERE result_id=?", [resultId]); db.run("DELETE FROM hierarchy_leaves WHERE result_id=?", [resultId]); db.run("DELETE FROM leaf_order WHERE result_id=?", [resultId]); db.run("DELETE FROM visualization_points WHERE result_id=?", [resultId]); db.run("DELETE FROM cluster_titles WHERE result_id=?", [resultId]); db.run("DELETE FROM soft_memberships WHERE result_id=?", [resultId]);
      result.ids.forEach((path, i) => db.run("INSERT INTO assignments(result_id,path,leaf_label,probability,outlier_score) VALUES(?,?,?,?,?)", [resultId, path, result.leafLabels[i], result.probabilities[i], result.outlierProxy[i]]));
      result.hierarchy.leaves.forEach((leaf, ordinal) => db.run("INSERT INTO hierarchy_leaves(result_id,ordinal,leaf_id) VALUES(?,?,?)", [resultId, ordinal, leaf]));
      (result.leafOrder || result.hierarchy.leaves).forEach((leaf, ordinal) => db.run("INSERT INTO leaf_order(result_id,ordinal,leaf_id) VALUES(?,?,?)", [resultId, ordinal, leaf]));
      result.hierarchy.merges.forEach((merge) => db.run("INSERT INTO hierarchy_merges(result_id,id,left_id,right_id,distance,mass) VALUES(?,?,?,?,?,?)", [resultId, merge.id, merge.left, merge.right, merge.distance, merge.mass]));
      result.visualization?.coordinates.forEach((point, ordinal) => db.run("INSERT INTO visualization_points(result_id,ordinal,path,x,y,leaf_label) VALUES(?,?,?,?,?,?)", [resultId, ordinal, result.ids[ordinal], point[0], point[1], result.visualization!.labels[ordinal] ?? result.leafLabels[ordinal]]));
      for (const [node, title] of Object.entries(result.titles || {})) db.run("INSERT INTO cluster_titles(result_id,node_id,title) VALUES(?,?,?)", [resultId, Number(node), title]);
      if (result.softMemberships) result.softMemberships.forEach((row, ordinal) => row.forEach((membership, leafIndex) => db.run("INSERT INTO soft_memberships(result_id,path,leaf_id,membership) VALUES(?,?,?,?)", [resultId, result.ids[ordinal], (result.leafOrder || result.hierarchy.leaves)[leafIndex], membership])));
      for (const [path, memberships] of Object.entries(options.softMemberships || {})) for (const [leaf, membership] of Object.entries(memberships)) db.run("INSERT INTO soft_memberships(result_id,path,leaf_id,membership) VALUES(?,?,?,?)", [resultId, path, Number(leaf), membership]);
    });
    return resultId;
  }
  async getResult(resultId?: string): Promise<ClusterResult | null> {
    this.requireOpen();
    const where = resultId ? ` WHERE result_id=${sqlQuote(resultId)}` : "";
    const rows = this.db.exec(`SELECT result_json FROM results${where} ORDER BY created_at DESC LIMIT 1`);
    return rows[0]?.values[0]?.[0] ? JSON.parse(String(rows[0].values[0][0])) as ClusterResult : null;
  }
  async getNote(path: string): Promise<NoteRecord | undefined> {
    this.requireOpen(); const rows = this.db.exec(`SELECT path,title,mtime,content_hash,content FROM notes WHERE path=${sqlQuote(path)}`); const row = rows[0]?.values[0];
    if (!row) return undefined; const object = Object.fromEntries(rows[0].columns.map((key, i) => [key, row[i]]));
    return { path: String(object.path), title: String(object.title), mtime: Number(object.mtime), hash: String(object.content_hash), content: object.content == null ? "" : String(object.content) };
  }
  async getPcaCoordinates(path: string, modelHash: string): Promise<number[] | undefined> {
    this.requireOpen(); const rows = this.db.exec(`SELECT coordinates_json FROM pca_coordinates WHERE path=${sqlQuote(path)} AND model_hash=${sqlQuote(modelHash)}`);
    return rows[0]?.values[0]?.[0] ? JSON.parse(String(rows[0].values[0][0])) : undefined;
  }
  getVisualization(resultId: string): Array<{ path: string; x: number; y: number; leafLabel: number }> {
    return this.query("SELECT path,x,y,leaf_label AS leafLabel FROM visualization_points WHERE result_id=? ORDER BY ordinal", [resultId]).map((row) => ({ path: String(row.path), x: Number(row.x), y: Number(row.y), leafLabel: Number(row.leafLabel) }));
  }
  getSoftMemberships(resultId: string): Array<{ path: string; leafId: number; membership: number }> {
    return this.query("SELECT path,leaf_id AS leafId,membership FROM soft_memberships WHERE result_id=? ORDER BY path,leaf_id", [resultId]).map((row) => ({ path: String(row.path), leafId: Number(row.leafId), membership: Number(row.membership) }));
  }
  query(sql: string, params: unknown[] = []): Array<Record<string, unknown>> {
    this.requireOpen(); const statement = this.db.prepare(sql); const result: Array<Record<string, unknown>> = [];
    try { statement.bind(params); while (statement.step()) result.push(statement.getAsObject()); } finally { statement.free(); } return result;
  }
  async saveEmbeddingLog(log: EmbeddingRunLog): Promise<void> { await this.transaction((db) => db.run("INSERT INTO embedding_logs(started_at,completed_at,provider,model,status,log_json) VALUES(?,?,?,?,?,?)", [log.startedAt, log.completedAt, log.provider, log.model, log.status || null, JSON.stringify(log)])); }
  async loadLatestEmbeddingLog(): Promise<EmbeddingRunLog | null> { this.requireOpen(); const rows = this.db.exec("SELECT log_json FROM embedding_logs ORDER BY id DESC LIMIT 1"); return rows[0]?.values[0]?.[0] ? JSON.parse(String(rows[0].values[0][0])) : null; }
}

/** Convenience factory for the async `initSqlJs({ locateFile })` API. */
export async function createSqliteStore(
  adapter: BinaryAdapter,
  initializer: SqlJsStatic | Promise<SqlJsStatic> | (() => Promise<SqlJsStatic>),
  options: SqliteStorageOptions = {}
): Promise<SqliteClusterStore> {
  const sql = typeof initializer === "function" ? await initializer() : await initializer;
  return new SqliteClusterStore(adapter, sql, options).open();
}

function sqlQuote(value: string): string { return `'${value.split("'").join("''")}'`; }

export interface LegacyJsonSources { embeddingCache?: string; result?: string; embeddingLog?: string; }
/** Import the old three JSON files exactly once. Invalid or absent files are ignored. */
export async function migrateLegacyJson(store: SqliteClusterStore, sources: LegacyJsonSources): Promise<{ migrated: string[] }> {
  const migrated: string[] = [];
  await store.transaction((db) => {
    const importOne = (name: string, raw: string | undefined, action: () => void) => {
      if (!raw || db.exec("SELECT 1 FROM migrations WHERE name=" + sqlQuote(name))[0]?.values.length) return;
      try { JSON.parse(raw); action(); db.run("INSERT INTO migrations(name,completed_at) VALUES(?,?)", [name, new Date().toISOString()]); migrated.push(name); } catch { /* leave it available for a later repair */ }
    };
    importOne("embedding-cache.json", sources.embeddingCache, () => {
      const document = JSON.parse(sources.embeddingCache!); for (const entry of document.embeddings || []) {
        // Legacy cache entries predate note metadata. A zero-metadata note is
        // upgraded in place when the vault is scanned next.
        db.run("INSERT OR IGNORE INTO notes(path,title,mtime,content_hash,content) VALUES(?,?,?,?,?)", [entry.path, entry.path.split("/").pop() || entry.path, 0, entry.hash || "legacy", null]);
        db.run("INSERT OR IGNORE INTO embeddings(path,provider,model,embedding_hash,note_content_hash,dimension,vector_json,created_at) VALUES(?,?,?,?,?,?,?,?)", [entry.path, entry.provider, entry.model, entry.hash, entry.hash, entry.vector.length, JSON.stringify(entry.vector), new Date().toISOString()]);
      }
    });
    importOne("cluster-result.json", sources.result, () => {
      const result = JSON.parse(sources.result!); const id = `legacy-${Date.now()}`; const timestamp = new Date().toISOString();
      db.run("INSERT OR IGNORE INTO results(result_id,schema_version,created_at,result_json) VALUES(?,?,?,?)", [id, result.schemaVersion || 3, timestamp, JSON.stringify(result)]);
      for (const [i, path] of (result.ids || []).entries()) {
        db.run("INSERT OR IGNORE INTO notes(path,title,mtime,content_hash,content) VALUES(?,?,?,?,?)", [path, String(path).split("/").pop() || path, 0, "legacy", null]);
        db.run("INSERT OR IGNORE INTO assignments(result_id,path,leaf_label,probability,outlier_score) VALUES(?,?,?,?,?)", [id, path, result.leafLabels?.[i] ?? -1, result.probabilities?.[i] ?? 0, result.outlierProxy?.[i] ?? 0]);
      }
      for (const [ordinal, leaf] of (result.hierarchy?.leaves || []).entries()) db.run("INSERT OR IGNORE INTO hierarchy_leaves(result_id,ordinal,leaf_id) VALUES(?,?,?)", [id, ordinal, leaf]);
      for (const merge of result.hierarchy?.merges || []) db.run("INSERT OR IGNORE INTO hierarchy_merges(result_id,id,left_id,right_id,distance,mass) VALUES(?,?,?,?,?,?)", [id, merge.id, merge.left, merge.right, merge.distance, merge.mass]);
      for (const [node, title] of Object.entries(result.titles || {})) db.run("INSERT OR IGNORE INTO cluster_titles(result_id,node_id,title) VALUES(?,?,?)", [id, Number(node), title]);
    });
    importOne("embedding-log.json", sources.embeddingLog, () => { const log = JSON.parse(sources.embeddingLog!); db.run("INSERT INTO embedding_logs(started_at,completed_at,provider,model,status,log_json) VALUES(?,?,?,?,?,?)", [log.startedAt || new Date().toISOString(), log.completedAt || new Date().toISOString(), log.provider || "unknown", log.model || "unknown", log.status || null, JSON.stringify(log)]); });
  });
  return { migrated };
}

/** Read legacy files from an Obsidian adapter and perform one-time migration. */
export async function migrateLegacyAdapter(store: SqliteClusterStore, adapter: { read(path: string): Promise<string> }): Promise<{ migrated: string[] }> {
  const read = async (path: string) => { try { return await adapter.read(path); } catch { return undefined; } };
  return migrateLegacyJson(store, { embeddingCache: await read(".obsidian/plugins/atomic-clusters/embedding-cache.json"), result: await read(".obsidian/plugins/atomic-clusters/cluster-result.json"), embeddingLog: await read(".obsidian/plugins/atomic-clusters/embedding-log.json") });
}

export { SCHEMA as SQLITE_SCHEMA };
