declare module "sql.js" {
  interface Statement {
    bind(values?: unknown[] | Record<string, unknown>): void;
    step(): boolean;
    getAsObject(): Record<string, unknown>;
    free(): void;
  }
  interface Database {
    run(sql: string, params?: unknown[] | Record<string, unknown>): void;
    prepare(sql: string): Statement;
    exec(sql: string): Array<{ columns: string[]; values: unknown[][] }>;
    export(): Uint8Array;
    close(): void;
  }
  interface SqlJsStatic { Database: new (data?: ArrayLike<number>) => Database; }
  const initSqlJs: (config?: { locateFile?: (file: string) => string }) => Promise<SqlJsStatic>;
  export default initSqlJs;
}
