/** Renderer-safe formatting for persistent long-operation progress notices. */

export const PROGRESS_HEARTBEAT_INTERVAL_MS = 10_000;
export const PROGRESS_HEARTBEAT_STALL_MS = 30_000;

export function formatElapsedMs(elapsedMs: number): string {
  const totalSeconds = Math.max(0, Math.floor((Number.isFinite(elapsedMs) ? elapsedMs : 0) / 1000));
  const hours = Math.floor(totalSeconds / 3600);
  const minutes = Math.floor((totalSeconds % 3600) / 60);
  const seconds = totalSeconds % 60;
  if (hours > 0) return `${hours}h ${String(minutes).padStart(2, "0")}m ${String(seconds).padStart(2, "0")}s`;
  if (minutes > 0) return `${minutes}m ${String(seconds).padStart(2, "0")}s`;
  return `${seconds}s`;
}

export function buildProgressHeartbeatDetail(baseDetail: string, elapsedMs: number, silentForMs: number, stallMs = PROGRESS_HEARTBEAT_STALL_MS): string {
  const prefix = baseDetail.trim() || "Working";
  const stalled = silentForMs >= stallMs ? " · Still working…" : "";
  return `${prefix} · ${formatElapsedMs(elapsedMs)} elapsed${stalled}`;
}
