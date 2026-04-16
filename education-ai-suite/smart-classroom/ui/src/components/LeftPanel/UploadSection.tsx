import React, { useRef, useState, useCallback, useEffect } from "react";
import { useTranslation } from "react-i18next";
import "../../assets/css/UploadSection.css";
import { csUploadIngest, csQueryTask, csIngest, csCleanupTask, csCheckHasData, createSession, startMonitoring, csGetConfig } from "../../services/api";
import { useAppDispatch, useAppSelector } from "../../redux/hooks";
import { setCsProcessing, setSessionId, setMonitoringActive, setCsUploadsComplete, setCsHasUploads, setCsDbHasData, addCsAvailableLabels } from "../../redux/slices/uiSlice";

type TaskStatus =
  | "STAGED"
  | "PENDING"
  | "PROCESSING"
  | "COMPLETED"
  | "FAILED"
  | "ALREADY_EXISTS";

type VideoSummaryStatus = "PROCESSING" | "COMPLETED" | "FAILED" | null;

interface UploadEntry {
  id: string;
  file: File;
  filename: string;
  fileType: string;
  fileSize: number;
  taskId: string | null;
  fileKey: string | null;
  status: TaskStatus;
  progress: number;
  error: string | null;
  selected: boolean;
  tags: string[];
  videoSummaryStatus: VideoSummaryStatus;
  isVideo: boolean;
  vsEnabled: boolean;
}

const POLL_INTERVAL_MS = 3000;

function genId() {
  return Math.random().toString(36).slice(2);
}

function formatSize(bytes: number): string {
  if (bytes < 1024) return `${bytes} B`;
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
  return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
}

const ALLOWED_EXTENSIONS = new Set([".mp4", ".ppt", ".pptx", ".docx", ".pdf", ".jpg", ".jpeg", ".csv", ".txt"]);
const VIDEO_EXTENSIONS = new Set([".mp4", ".avi", ".mov", ".mkv"]);

function isAllowed(filename: string): boolean {
  const ext = filename.slice(filename.lastIndexOf(".")).toLowerCase();
  return ALLOWED_EXTENSIONS.has(ext);
}

function isVideoFile(filename: string): boolean {
  const ext = filename.slice(filename.lastIndexOf(".")).toLowerCase();
  return VIDEO_EXTENSIONS.has(ext);
}

const TERMINAL: TaskStatus[] = ["COMPLETED", "FAILED", "ALREADY_EXISTS"];
const ACTIVE: TaskStatus[] = ["PROCESSING", "PENDING"];

const UploadSection: React.FC = () => {
  const { t } = useTranslation();
  const dispatch = useAppDispatch();
  const sessionId = useAppSelector((s) => s.ui.sessionId);
  const monitoringActive = useAppSelector((s) => s.ui.monitoringActive);
  const sessionIdRef = useRef<string | null>(sessionId);
  const monitoringActiveRef = useRef<boolean>(monitoringActive);
  useEffect(() => { sessionIdRef.current = sessionId; }, [sessionId]);
  useEffect(() => { monitoringActiveRef.current = monitoringActive; }, [monitoringActive]);
  const fileInputRef = useRef<HTMLInputElement>(null);

  const [entries, setEntries] = useState<UploadEntry[]>([]);
  const [isDragOver, setIsDragOver] = useState(false);
  const [confirmRemoveId, setConfirmRemoveId] = useState<string | null>(null);
  const [summarizationEnabled, setSummarizationEnabled] = useState(false);

  useEffect(() => {
    csGetConfig().then((cfg) => {
      setSummarizationEnabled(cfg.video_summarization_enabled === true);
    });
  }, []);

  const selectAllRef = useRef<HTMLInputElement>(null);
  const stagedEntries = entries.filter((e) => e.status === "STAGED");
  const allSelected = stagedEntries.length > 0 && stagedEntries.every((e) => e.selected);
  const someSelected = stagedEntries.some((e) => e.selected);

  useEffect(() => {
    if (selectAllRef.current) {
      selectAllRef.current.indeterminate = someSelected && !allSelected;
    }
  }, [someSelected, allSelected]);

  // Track uploads status for search section — only set true reactively;
  // false is set explicitly on user-initiated clear to avoid resetting
  // SearchSection due to transient remounts (StrictMode or navigation).
  useEffect(() => {
    if (entries.length > 0) {
      dispatch(setCsHasUploads(true));
      const completedEntries = entries.filter(
        (e) => e.status === "COMPLETED"
      );
      if (completedEntries.length > 0 || entries.some((e) => e.status === "ALREADY_EXISTS")) {
        dispatch(setCsUploadsComplete(true));
        dispatch(setCsDbHasData(true));
      }
      // Only add labels to filter when files are successfully uploaded (COMPLETED)
      const completedLabels = completedEntries.flatMap((e) => e.tags);
      if (completedLabels.length > 0) {
        dispatch(addCsAvailableLabels(completedLabels));
      }
    }
  }, [entries, dispatch]);

  const toggleSelectAll = () => {
    const next = !allSelected;
    setEntries((prev) => prev.map((e) => e.status === "STAGED" ? { ...e, selected: next } : e));
  };

  const toggleSelect = (id: string) => {
    setEntries((prev) =>
      prev.map((e) => (e.id === id && e.status === "STAGED" ? { ...e, selected: !e.selected } : e))
    );
  };

  // ── Tag editor ──────────────────────────────────────────────
  const [tagInput, setTagInput] = useState("");

  const selectedEntries = entries.filter((e) => e.selected);

  // Parse input text into unique tags (one label per line)
  const parseTags = (text: string): string[] =>
    [...new Set(text.split(/\n/).map((s) => s.trim()).filter(Boolean))];

  // "Add to selected" — parse input and apply to selected (STAGED only) files
  const applyTagsToSelected = () => {
    const tags = parseTags(tagInput);
    if (tags.length === 0 || selectedEntries.length === 0) return;
    setEntries((prev) =>
      prev.map((e) => {
        if (!e.selected || e.status !== "STAGED") return e;
        const newTags = tags.filter((t) => !e.tags.includes(t));
        if (newTags.length === 0) return e;
        return { ...e, tags: [...e.tags, ...newTags] };
      })
    );
    setTagInput("");
  };

  const removeTag = (entryId: string, tag: string) => {
    setEntries((prev) =>
      prev.map((e) => {
        if (e.id !== entryId || e.status !== "STAGED") return e;
        return { ...e, tags: e.tags.filter((t) => t !== tag) };
      })
    );
  };

  const pollTimers = useRef<Record<string, ReturnType<typeof setInterval>>>({}); 

  // Drive csProcessing flag: true while any entry is actively uploading/processing
  useEffect(() => {
    const anyActive = entries.some((e) => ACTIVE.includes(e.status));
    dispatch(setCsProcessing(anyActive));
  }, [entries, dispatch]);

  const updateEntry = useCallback(
    (id: string, patch: Partial<UploadEntry>) => {
      setEntries((prev) =>
        prev.map((e) => (e.id === id ? { ...e, ...patch } : e))
      );
    },
    []
  );

  const startPolling = useCallback(
    (entryId: string, taskId: string) => {
      const timer = setInterval(async () => {
        try {
          const result = await csQueryTask(taskId);
          let status = (result.status?.toUpperCase() ?? "PROCESSING") as TaskStatus;
          const progress =
            status === "COMPLETED"
              ? 100
              : typeof result.progress === "number"
              ? result.progress
              : 0;

          const fileKey =
            (result.result?.file_info as any)?.file_key ??
            (result.result as any)?.file_key ??
            null;

          const vss = (result.result as any)?.video_summary_status as VideoSummaryStatus ?? null;
          updateEntry(entryId, { status, progress, videoSummaryStatus: vss, ...(fileKey ? { fileKey } : {}) });

          // Stop polling when task is terminal AND no video summary is still processing
          const isTerminal = status === "COMPLETED" || status === "FAILED";
          const summaryDone = vss !== "PROCESSING";
          if (isTerminal && summaryDone) {
            clearInterval(pollTimers.current[entryId]);
            delete pollTimers.current[entryId];
          }
        } catch {
          // ignore transient poll errors
        }
      }, POLL_INTERVAL_MS);

      pollTimers.current[entryId] = timer;
    },
    [updateEntry]
  );

  // Stage files locally — upload is triggered explicitly by the user
  const processFiles = useCallback(
    (files: File[]) => {
      setEntries((prev) => {
        // Deduplicate by filename + size against existing entries
        const existingKeys = new Set(prev.map((e) => `${e.filename}|${e.fileSize}`));
        const unique = files.filter((f) => !existingKeys.has(`${f.name}|${f.size}`));
        if (unique.length === 0) return prev;

        const newEntries: UploadEntry[] = unique.map((f) => ({
          id: genId(),
          file: f,
          filename: f.name,
          fileType: f.name.split(".").pop()?.toUpperCase() ?? "—",
          fileSize: f.size,
          taskId: null,
          fileKey: null,
          status: "STAGED" as TaskStatus,
          progress: 0,
          error: null,
          selected: false,
          tags: [],
          videoSummaryStatus: null,
          isVideo: isVideoFile(f.name),
          vsEnabled: true,
        }));
        return [...prev, ...newEntries];
      });
    },
    []
  );

  // Upload all staged files (with their tags) when user clicks the Upload button
  const handleUploadAll = useCallback(async () => {
    // Auto-apply any pending labels from the input before uploading
    const pendingTags = parseTags(tagInput);
    let entriesToUpload = entries;
    if (pendingTags.length > 0) {
      entriesToUpload = entries.map((e) => {
        if (!e.selected || e.status !== "STAGED") return e;
        const newTags = pendingTags.filter((t) => !e.tags.includes(t));
        if (newTags.length === 0) return e;
        return { ...e, tags: [...e.tags, ...newTags] };
      });
      setTagInput("");
    }

    const stagedEntries = entriesToUpload.filter((e) => e.status === "STAGED");
    if (!stagedEntries.length) return;

    // Update entries state with any auto-applied tags, mark as PROCESSING
    setEntries(
      entriesToUpload.map((e) =>
        e.status === "STAGED" ? { ...e, status: "PROCESSING" as TaskStatus, selected: false } : e
      )
    );

    // Ensure session + monitoring without blocking uploads
    const ensureSessionAndMonitoring = async () => {
      if (!sessionIdRef.current) {
        try {
          const res = await createSession();
          sessionIdRef.current = res.sessionId;
          dispatch(setSessionId(res.sessionId));
        } catch (e) {
          console.warn("Could not create session for metrics:", e);
        }
      }
      if (sessionIdRef.current && !monitoringActiveRef.current) {
        try {
          await startMonitoring(sessionIdRef.current);
          dispatch(setMonitoringActive(true));
        } catch (e) {
          console.warn("Could not start monitoring:", e);
        }
      }
    };
    ensureSessionAndMonitoring();

    await Promise.all(
      stagedEntries.map(async (entry) => {
        try {
          const meta: Record<string, unknown> = {};
          if (entry.tags.length) meta.tags = entry.tags;
          if (entry.isVideo) meta.vs_enabled = entry.vsEnabled;
          // If file already exists on server (re-staged with new tags), re-ingest with updated tags
          // Otherwise do a fresh upload+ingest
          if (entry.fileKey) {
            const ingestRes = await csIngest(entry.fileKey, meta);
            updateEntry(entry.id, { taskId: ingestRes.task_id, status: "PROCESSING", fileKey: entry.fileKey });
            startPolling(entry.id, ingestRes.task_id);
          } else {
            const res = await csUploadIngest(entry.file, Object.keys(meta).length ? meta : undefined);
            if (res.status === "ALREADY_EXISTS") {
              // File was already fully processed in a previous session — no new task created
              updateEntry(entry.id, { status: "ALREADY_EXISTS", progress: 100 });
            } else {
              updateEntry(entry.id, { taskId: res.task_id, status: "PROCESSING", fileKey: res.file_key ?? null });
              startPolling(entry.id, res.task_id);
            }
          }
        } catch (err: any) {
          updateEntry(entry.id, {
            status: "FAILED",
            error: err?.message ?? "Upload failed",
          });
        }
      })
    );
  }, [entries, tagInput, updateEntry, startPolling, dispatch]);

  const handleRetry = useCallback(
    async (entry: UploadEntry) => {
      updateEntry(entry.id, { status: "PROCESSING", progress: 0, error: null, taskId: null });
      try {
        const meta: Record<string, unknown> = {};
        if (entry.tags.length) meta.tags = entry.tags;
        if (entry.isVideo) meta.vs_enabled = entry.vsEnabled;
        const res = await csUploadIngest(entry.file, Object.keys(meta).length ? meta : undefined);
        if (res.status === "ALREADY_EXISTS") {
          // Already fully processed — no new background task, treat as terminal
          updateEntry(entry.id, { status: "ALREADY_EXISTS", progress: 100 });
        } else {
          updateEntry(entry.id, { taskId: res.task_id, status: "PROCESSING", fileKey: res.file_key ?? null });
          startPolling(entry.id, res.task_id);
        }
      } catch (err: any) {
        updateEntry(entry.id, { status: "FAILED", error: err?.message ?? "Upload failed" });
      }
    },
    [updateEntry, startPolling]
  );

  const handleBrowse = () => fileInputRef.current?.click();

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const files = Array.from(e.target.files ?? []).filter((f) => isAllowed(f.name));
    if (files.length) processFiles(files);
    if (fileInputRef.current) fileInputRef.current.value = "";
  };

  const handleDragOver = (e: React.DragEvent) => {
    e.preventDefault();
    setIsDragOver(true);
  };

  const handleDragLeave = () => setIsDragOver(false);

  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault();
    setIsDragOver(false);
    const files = Array.from(e.dataTransfer.files).filter((f) => isAllowed(f.name));
    if (files.length) processFiles(files);
  };

  const confirmRemove = () => {
    const id = confirmRemoveId;
    if (!id) return;

    // Stop polling
    if (pollTimers.current[id]) {
      clearInterval(pollTimers.current[id]);
      delete pollTimers.current[id];
    }

    // Read the entry synchronously before setEntries batches the update
    const removedEntry = entries.find((e) => e.id === id);

    // Remove from UI
    setEntries((prev) => {
      const next = prev.filter((e) => e.id !== id);
      if (next.length === 0) {
        // Last file removed — re-check if backend DB still has data
        csCheckHasData().then((hasData) => {
          dispatch(setCsDbHasData(hasData));
          dispatch(setCsHasUploads(hasData));
          dispatch(setCsUploadsComplete(hasData));
        }).catch(() => {
          dispatch(setCsDbHasData(false));
          dispatch(setCsHasUploads(false));
          dispatch(setCsUploadsComplete(false));
        });
      }
      return next;
    });
    setConfirmRemoveId(null);

    // Call backend cleanup if the file was uploaded (has a valid taskId)
    if (removedEntry?.taskId?.trim()) {
      csCleanupTask(removedEntry.taskId).catch((err) =>
        console.warn(`Cleanup failed for task ${removedEntry!.taskId}:`, err)
      );
    }
  };

  const getStatusLabel = (s: TaskStatus) => {
    switch (s) {
      case "STAGED":         return t("uploadSection.staged");
      case "PENDING":        return t("uploadSection.pending");
      case "PROCESSING":     return t("uploadSection.processing");
      case "COMPLETED":      return t("uploadSection.uploaded");
      case "FAILED":         return t("uploadSection.failed");
      case "ALREADY_EXISTS": return t("uploadSection.alreadyExists");
    }
  };

return (
  <>
    <div className="cs-upload-card">
      <div className="cs-upload-header">
        <span className="cs-upload-title">{t("uploadSection.upload")}</span>
      </div>

      <div
        className={`cs-dropzone-modern ${isDragOver ? "cs-dropzone-modern--active" : ""}`}
        onDragOver={handleDragOver}
        onDragLeave={handleDragLeave}
        onDrop={handleDrop}
        onClick={handleBrowse}
      >
        <div className="cs-upload-icon">⇪</div>
        <p className="cs-upload-main-text">{t("uploadSection.dragDrop")}</p>
        <p className="cs-upload-link-text">{t("uploadSection.orClick")}</p>
      </div>

      <p className="cs-supported-types">{t("uploadSection.supportedTypes")}</p>
      <p className="cs-max-size-hint">{t("uploadSection.maxTotalSize", "Max total size: 500 MB")}</p>

      <input
        ref={fileInputRef}
        type="file"
        multiple
        accept=".mp4,.ppt,.pptx,.docx,.pdf,.jpg,.jpeg,.csv,.txt"
        style={{ display: "none" }}
        onChange={handleFileChange}
      />

      {/* ── Labels: hint when no files selected, editor when files selected ── */}
      {stagedEntries.length > 0 && (
        selectedEntries.length > 0 ? (
          <div className="cs-labels-panel">
            <div className="cs-labels-title">{t("uploadSection.labelsTitle", "Labels")}</div>
            <div className="cs-labels-input-row">
              <textarea
                className="cs-labels-input cs-labels-textarea"
                placeholder={t("uploadSection.enterTagsPlaceholder", "Press Enter to separate multiple labels, e.g. Math English")}
                value={tagInput}
                onChange={(e) => setTagInput(e.target.value)}
                rows={3}
              />
              <button
                className="cs-labels-add-btn"
                disabled={parseTags(tagInput).length === 0 || selectedEntries.length === 0}
                onClick={applyTagsToSelected}
              >
                {t("uploadSection.addToSelected", "Add to selected")}
              </button>
            </div>
          </div>
        ) : (
          <p className="cs-labels-hint">{t("uploadSection.selectFileToAddTags")}</p>
        )
      )}

      {/* ── File Table ── */}
      {entries.length > 0 && (
        <>
          <table className="cs-file-table">
            <thead>
              <tr>
                <th className="cs-col-check">
                  <input
                    ref={selectAllRef}
                    type="checkbox"
                    checked={allSelected}
                    onChange={toggleSelectAll}
                    disabled={stagedEntries.length === 0}
                    className="cs-checkbox"
                  />
                </th>
                <th>{t("uploadSection.fileName")}</th>
                <th>{t("uploadSection.type")}</th>
                <th>{t("uploadSection.size")}</th>
                <th>{t("uploadSection.status")}</th>
                {summarizationEnabled && entries.some((e) => e.isVideo) && (
                  <th className="cs-col-vs">{t("uploadSection.summarize")}</th>
                )}
                <th></th>
              </tr>
            </thead>
            <tbody>
              {entries.map((entry) => (
                <tr
                  key={entry.id}
                  className={`cs-row-${entry.status.toLowerCase()}${entry.selected ? " cs-row-selected" : ""}`}
                >
                  <td>
                    <input
                      type="checkbox"
                      checked={entry.selected}
                      onChange={() => toggleSelect(entry.id)}
                      disabled={entry.status !== "STAGED"}
                      className="cs-checkbox"
                    />
                  </td>
                  <td>
                    <span className="cs-file-name" title={entry.filename}>
                      {entry.filename}
                    </span>
                    {entry.tags.length > 0 && (
                      <div className="cs-row-tags">
                        {entry.tags.map((tg) => (
                          <span key={tg} className="cs-row-chip">
                            {tg}
                            {entry.status === "STAGED" && (
                              <button
                                className="cs-row-chip-remove"
                                onClick={(e) => { e.stopPropagation(); removeTag(entry.id, tg); }}
                              >✕</button>
                            )}
                          </span>
                        ))}
                      </div>
                    )}
                  </td>
                  <td>{entry.fileType}</td>
                  <td>{formatSize(entry.fileSize)}</td>
                  <td className="cs-col-status">
                    {entry.status === "FAILED" ? (
                      <div className="cs-failed-cell">
                        <span className="cs-failed-msg" title={entry.error ?? ""}>
                          Upload of &apos;{entry.filename}&apos; failed. Please try again
                        </span>
                        <div className="cs-failed-actions">
                          <button
                            className="cs-retry-btn"
                            onClick={() => handleRetry(entry)}
                          >
                            {t("uploadSection.retry")}
                          </button>
                          <button
                            className="cs-retry-btn cs-retry-btn--remove"
                            onClick={() => setConfirmRemoveId(entry.id)}
                          >
                            {t("uploadSection.remove")}
                          </button>
                        </div>
                      </div>
                    ) : (
                      <span
                        className={`cs-status-badge cs-status-badge--${entry.status.toLowerCase()}`}
                      >
                        {getStatusLabel(entry.status)}
                      </span>
                    )}
                  </td>
                  {summarizationEnabled && entries.some((e) => e.isVideo) && (
                    <td className="cs-col-vs">
                      {entry.isVideo ? (
                        entry.status === "STAGED" ? (
                          <label className="cs-vs-toggle" onClick={(e) => e.stopPropagation()}>
                            <input
                              type="checkbox"
                              checked={entry.vsEnabled}
                              onChange={() => updateEntry(entry.id, { vsEnabled: !entry.vsEnabled })}
                              className="cs-checkbox"
                            />
                          </label>
                        ) : entry.videoSummaryStatus === "PROCESSING" ? (
                          <span className="cs-status-badge cs-status-badge--summarizing">
                            <span className="cs-spinner" />
                            {t("uploadSection.summarizing")}
                          </span>
                        ) : entry.videoSummaryStatus === "COMPLETED" ? (
                          <span className="cs-status-badge cs-status-badge--summarized">
                            {t("uploadSection.summarized")}
                          </span>
                        ) : entry.videoSummaryStatus === "FAILED" ? (
                          <span className="cs-status-badge cs-status-badge--summary-failed">
                            {t("uploadSection.summaryFailed")}
                          </span>
                        ) : entry.status === "COMPLETED" && !entry.vsEnabled ? (
                          <span className="cs-vs-off">{t("uploadSection.vsOff")}</span>
                        ) : null
                      ) : (
                        <span className="cs-vs-na">—</span>
                      )}
                    </td>
                  )}
                  <td className="cs-col-remove">
                    {entry.status !== "FAILED" && (
                      <button
                        className="cs-remove-btn"
                        disabled={ACTIVE.includes(entry.status) || entry.videoSummaryStatus === "PROCESSING"}
                        onClick={() => setConfirmRemoveId(entry.id)}
                        title={ACTIVE.includes(entry.status) || entry.videoSummaryStatus === "PROCESSING" ? "Cannot remove while processing" : "Remove file"}
                      >
                        🗑
                      </button>
                    )}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>

          <div className="cs-table-footer">
            <button
              className="cs-clear-all-btn"
              disabled={entries.some((e) => ACTIVE.includes(e.status))}
              onClick={() => {
                Object.values(pollTimers.current).forEach(clearInterval);
                pollTimers.current = {};
                setEntries([]);
                // Re-check if backend DB still has searchable data
                csCheckHasData().then((hasData) => {
                  dispatch(setCsDbHasData(hasData));
                  dispatch(setCsHasUploads(hasData));
                  dispatch(setCsUploadsComplete(hasData));
                }).catch(() => {
                  dispatch(setCsDbHasData(false));
                  dispatch(setCsHasUploads(false));
                  dispatch(setCsUploadsComplete(false));
                });
              }}
            >
              {t("uploadSection.clearAll")}
            </button>
            <button
              className="cs-upload-all-btn"
              disabled={!entries.some((e) => e.status === "STAGED")}
              onClick={handleUploadAll}
            >
              {t("uploadSection.uploadFiles")}
            </button>
          </div>
        </>
      )}
    </div>

    {confirmRemoveId && (
      <div className="cs-modal-overlay">
        <div className="cs-modal">
          <p>{t("uploadSection.removeFileConfirmation")}</p>
          <div className="cs-modal-actions">
            <button onClick={() => setConfirmRemoveId(null)}>{t("uploadSection.cancel")}</button>
            <button className="cs-danger-btn" onClick={confirmRemove}>
              {t("uploadSection.remove")}
            </button>
          </div>
        </div>
      </div>
    )}
  </>
);}

export default UploadSection;
