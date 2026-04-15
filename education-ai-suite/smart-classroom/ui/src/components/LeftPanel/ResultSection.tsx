import React, { useState, useMemo, useRef, useEffect } from "react";
import { useTranslation } from "react-i18next";
import "../../assets/css/ResultSection.css";
import searchIcon from "../../assets/images/search-icon.svg";
import { csDownloadUrl, extractFileKey } from "../../services/api";

// Content Search API types
export interface CsSearchParams {
  query?: string;
  image_base64?: string;
  max_num_results?: number;
  filter?: Record<string, string[]>;
}

export interface CsSearchResultMeta {
  file_name?: string;
  file_path?: string;
  type?: string;
  video_pin_second?: number;
  start_time?: number;
  end_time?: number;
  doc_page_number?: number;
  tags?: string[];
  doc_filetype?: string;
  summary_text?: string;
  chunk_text?: string;
}

export interface CsSearchResult {
  id: string;
  distance: number;
  meta: CsSearchResultMeta;
  score: number;
}

export type SearchResult = CsSearchResult;

type ResultTab = "all" | "document" | "image" | "video";

interface ResultSectionProps {
  results: SearchResult[];
}

function getFileName(result: SearchResult): string {
  const meta = result?.meta;
  if (!meta) return "Unknown";
  if (meta.file_name) return meta.file_name;
  if (meta.file_path) return meta.file_path.split("/").pop() || "Unknown";
  return "Unknown";
}

function fileExtension(name: string): string {
  const lower = (name || "").toLowerCase();
  if (!lower.includes(".")) return "";
  return lower.slice(lower.lastIndexOf("."));
}

function formatScore(score: number): string {
  if (score <= 0) return "0%";
  if (score < 1) return "< 1%";
  return `${Math.round(score)}%`;
}

function formatTimestamp(totalSeconds: number): string {
  const h = Math.floor(totalSeconds / 3600);
  const m = Math.floor((totalSeconds % 3600) / 60);
  const s = Math.floor(totalSeconds % 60);
  if (h > 0)
    return `${h}:${String(m).padStart(2, "0")}:${String(s).padStart(2, "0")}`;
  return `${String(m).padStart(2, "0")}:${String(s).padStart(2, "0")}`;
}

// ── Video Thumbnail ──────────────────────────────────────────

const VideoThumbnail: React.FC<{ url: string; seekTime: number }> = ({
  url,
  seekTime,
}) => {
  const imgRef = useRef<HTMLImageElement>(null);
  const [loaded, setLoaded] = useState(false);

  useEffect(() => {
    let active = true;
    const video = document.createElement("video");
    video.muted = true;
    video.preload = "auto";

    const cleanup = () => {
      video.removeAttribute("src");
      video.load();
    };

    video.addEventListener("loadedmetadata", () => {
      video.currentTime = Math.max(
        0,
        Math.min(seekTime || 1, video.duration - 0.1)
      );
    });

    video.addEventListener("seeked", () => {
      try {
        const canvas = document.createElement("canvas");
        const w = video.videoWidth;
        const h = video.videoHeight;
        const maxW = 300;
        const scale = Math.min(1, maxW / w);
        canvas.width = Math.round(w * scale);
        canvas.height = Math.round(h * scale);
        canvas
          .getContext("2d")
          ?.drawImage(video, 0, 0, canvas.width, canvas.height);
        if (active && imgRef.current) {
          imgRef.current.src = canvas.toDataURL("image/jpeg", 0.75);
          setLoaded(true);
        }
      } catch (e) {
        console.warn("Video thumbnail capture failed:", e);
      }
      cleanup();
    });

    video.addEventListener("error", cleanup);

    const timer = setTimeout(() => {
      if (active && !loaded) cleanup();
    }, 8000);

    video.src = url;

    return () => {
      active = false;
      clearTimeout(timer);
      cleanup();
    };
  }, [url, seekTime]);

  return (
    <>
      <img
        ref={imgRef}
        className={`cs-thumb-img ${loaded ? "" : "cs-thumb-hidden"}`}
        alt="Video thumbnail"
      />
      {!loaded && <div className="cs-thumb-shimmer" />}
    </>
  );
};

// ── PDF Thumbnail ────────────────────────────────────────────

const PdfThumbnail: React.FC<{ url: string; pageNum: number }> = ({
  url,
  pageNum,
}) => {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const [loaded, setLoaded] = useState(false);

  useEffect(() => {
    let cancelled = false;

    (async () => {
      try {
        const pdfjsLib = await import("pdfjs-dist");
        pdfjsLib.GlobalWorkerOptions.workerSrc = `https://cdnjs.cloudflare.com/ajax/libs/pdf.js/${pdfjsLib.version}/pdf.worker.min.js`;

        const pdf = await pdfjsLib.getDocument(url).promise;
        const safePage = Math.min(Math.max(1, pageNum), pdf.numPages);
        const page = await pdf.getPage(safePage);
        const vp = page.getViewport({ scale: 1 });
        const scale = 300 / vp.width;
        const scaled = page.getViewport({ scale });

        const canvas = canvasRef.current;
        if (!canvas || cancelled) return;
        canvas.width = scaled.width;
        canvas.height = scaled.height;
        const ctx = canvas.getContext("2d");
        if (ctx)
          await page.render({ canvasContext: ctx, viewport: scaled }).promise;
        if (!cancelled) setLoaded(true);
      } catch (err) {
        console.warn("PDF thumbnail failed:", err);
      }
    })();

    return () => {
      cancelled = true;
    };
  }, [url, pageNum]);

  return (
    <>
      <canvas
        ref={canvasRef}
        className={`cs-thumb-canvas ${loaded ? "" : "cs-thumb-hidden"}`}
      />
      {!loaded && <div className="cs-thumb-shimmer" />}
    </>
  );
};

// ── File Type Badge ──────────────────────────────────────────

const FILE_TYPE_PALETTE: Record<
  string,
  { bg: string; color: string; label: string }
> = {
  PDF: { bg: "#fee2e2", color: "#dc2626", label: "PDF" },
  DOCX: { bg: "#dbeafe", color: "#2563eb", label: "DOC" },
  DOC: { bg: "#dbeafe", color: "#2563eb", label: "DOC" },
  PPTX: { bg: "#fef3c7", color: "#d97706", label: "PPT" },
  PPT: { bg: "#fef3c7", color: "#d97706", label: "PPT" },
  XLSX: { bg: "#d1fae5", color: "#059669", label: "XLS" },
  XLS: { bg: "#d1fae5", color: "#059669", label: "XLS" },
  CSV: { bg: "#d1fae5", color: "#059669", label: "CSV" },
  TXT: { bg: "#f3f4f6", color: "#6b7280", label: "TXT" },
  HTML: { bg: "#fce7f3", color: "#db2777", label: "HTML" },
  MD: { bg: "#f3f4f6", color: "#6b7280", label: "MD" },
  JPG: { bg: "#ecfdf5", color: "#059669", label: "IMG" },
  JPEG: { bg: "#ecfdf5", color: "#059669", label: "IMG" },
  PNG: { bg: "#ecfdf5", color: "#059669", label: "IMG" },
  MP4: { bg: "#ede9fe", color: "#7c3aed", label: "VIDEO" },
};

const FileTypeBadge: React.FC<{ filename: string }> = ({ filename }) => {
  const ext = fileExtension(filename).replace(".", "").toUpperCase();
  const info = FILE_TYPE_PALETTE[ext] || {
    bg: "#f3f4f6",
    color: "#6b7280",
    label: ext || "FILE",
  };
  return (
    <div
      className="cs-thumb-badge"
      style={{ background: info.bg, color: info.color }}
    >
      <svg
        width="24"
        height="24"
        viewBox="0 0 24 24"
        fill="none"
        stroke="currentColor"
        strokeWidth="1.5"
        strokeLinecap="round"
        strokeLinejoin="round"
      >
        <path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z" />
        <polyline points="14 2 14 8 20 8" />
      </svg>
      <span className="cs-thumb-badge-label">{info.label}</span>
    </div>
  );
};

// ═════════════════════════════════════════════════════════════
// Preview Modal
// ═════════════════════════════════════════════════════════════

const PreviewModal: React.FC<{
  result: SearchResult;
  onClose: () => void;
}> = ({ result, onClose }) => {
  const meta = result?.meta || {};
  const fileName = meta.file_name || getFileName(result);
  const fileType = meta.type;
  const filePath = meta.file_path;
  const fileKey = filePath ? extractFileKey(filePath) : null;
  const downloadUrl = fileKey ? csDownloadUrl(fileKey) : null;
  const videoRef = useRef<HTMLVideoElement>(null);

  useEffect(() => {
    const handler = (e: KeyboardEvent) => {
      if (e.key === "Escape") onClose();
    };
    document.addEventListener("keydown", handler);
    document.body.style.overflow = "hidden";
    return () => {
      document.removeEventListener("keydown", handler);
      document.body.style.overflow = "";
    };
  }, [onClose]);

  const handleVideoLoaded = () => {
    const video = videoRef.current;
    if (video && meta.video_pin_second) {
      video.currentTime = meta.video_pin_second;
    }
  };

  const renderContent = () => {
    if (!downloadUrl)
      return <p className="cs-preview-empty">Preview not available</p>;

    if (fileType === "video") {
      const startTime = meta.video_pin_second || 0;
      return (
        <video
          ref={videoRef}
          className="cs-preview-video"
          controls
          autoPlay
          src={`${downloadUrl}#t=${startTime}`}
          onLoadedMetadata={handleVideoLoaded}
        />
      );
    }

    if (fileType === "image") {
      return (
        <img
          className="cs-preview-image"
          src={downloadUrl}
          alt={fileName}
        />
      );
    }

    if (fileType === "document" && meta.doc_filetype?.includes("pdf")) {
      return (
        <iframe
          className="cs-preview-pdf"
          src={downloadUrl}
          title={fileName}
        />
      );
    }

    if (meta.summary_text || meta.chunk_text) {
      return (
        <pre className="cs-preview-text">
          {meta.summary_text || meta.chunk_text}
        </pre>
      );
    }

    return (
      <div className="cs-preview-download">
        <p>Preview not available for this file type.</p>
        <a href={downloadUrl} download>
          Download {fileName}
        </a>
      </div>
    );
  };

  return (
    <div className="cs-preview-overlay" onClick={onClose}>
      <div className="cs-preview-modal" onClick={(e) => e.stopPropagation()}>
        <div className="cs-preview-header">
          <h3 className="cs-preview-title">{fileName}</h3>
          <button className="cs-preview-close" onClick={onClose}>
            &times;
          </button>
        </div>

        <div className="cs-preview-info-bar">
          <span>Type: {fileType}</span>
          <span>Score: {formatScore(result?.score ?? 0)}</span>
          {fileType === "video" && meta.video_pin_second != null && (
            <span>
              Match: {formatTimestamp(Math.floor(meta.video_pin_second))}
            </span>
          )}
          {fileType === "document" && meta.doc_page_number != null && (
            <span>Page: {meta.doc_page_number}</span>
          )}
        </div>

        <div className="cs-preview-body">{renderContent()}</div>
      </div>
    </div>
  );
};

// ═════════════════════════════════════════════════════════════
// ResultCard
// ═════════════════════════════════════════════════════════════

const ResultCard: React.FC<{
  result: SearchResult;
  onPreview: () => void;
}> = ({ result, onPreview }) => {
  const { t } = useTranslation();
  const meta = result?.meta || {};
  const fileName = meta.file_name || getFileName(result);
  const tags = Array.isArray(meta.tags) ? meta.tags : [];
  const fileType = meta.type;
  const filePath = meta.file_path;
  const fileKey = filePath ? extractFileKey(filePath) : null;
  const downloadUrl = fileKey ? csDownloadUrl(fileKey) : null;

  const renderThumbnail = () => {
    if (fileType === "image" && downloadUrl) {
      return (
        <img
          src={downloadUrl}
          alt={fileName}
          className="cs-thumb-img"
          loading="lazy"
        />
      );
    }

    if (fileType === "video" && downloadUrl) {
      return (
        <>
          <VideoThumbnail
            url={downloadUrl}
            seekTime={meta.video_pin_second || 0}
          />
          <div className="cs-thumb-play">
            <svg width="20" height="20" viewBox="0 0 24 24" fill="white">
              <path d="M8 5v14l11-7z" />
            </svg>
          </div>
          {meta.video_pin_second != null && (
            <span className="cs-thumb-time">
              {formatTimestamp(Math.floor(meta.video_pin_second))}
            </span>
          )}
        </>
      );
    }

    if (
      fileType === "document" &&
      meta.doc_filetype?.includes("pdf") &&
      downloadUrl
    ) {
      return (
        <PdfThumbnail
          url={downloadUrl}
          pageNum={meta.doc_page_number || 1}
        />
      );
    }

    return <FileTypeBadge filename={fileName} />;
  };

  return (
    <div className="cs-result-item" onClick={onPreview}>
      {/* Thumbnail */}
      <div className="cs-result-item-preview">{renderThumbnail()}</div>

      {/* Content */}
      <div className="cs-result-item-content">
        <div className="cs-result-item-row">
          <span className="cs-result-item-value" title={fileName}>
            {fileName}
          </span>
        </div>

        {/* Summary snippet */}
        {(meta.summary_text || meta.chunk_text) && (
          <div className="cs-result-item-summary">
            {(meta.summary_text || meta.chunk_text || "").slice(0, 100)}
            {(meta.summary_text || meta.chunk_text || "").length > 100
              ? "\u2026"
              : ""}
          </div>
        )}

        {/* Metadata: timestamp or page */}
        {fileType === "video" && meta.video_pin_second != null && (
          <div className="cs-result-item-meta">
            Time: {formatTimestamp(Math.floor(meta.video_pin_second))}
          </div>
        )}
        {fileType === "document" && meta.doc_page_number != null && (
          <div className="cs-result-item-meta">
            Page: {meta.doc_page_number}
          </div>
        )}

        {tags.length > 0 && (
          <div className="cs-result-item-row">
            <span className="cs-result-item-label">
              {t("resultSection.labels")}:
            </span>
            <div className="cs-result-item-tags">
              {tags.map((tag) => (
                <span key={tag} className="cs-result-item-tag">
                  {tag}
                </span>
              ))}
            </div>
          </div>
        )}
      </div>

      {/* Score */}
      <div className="cs-result-item-score-section">
        <span className="cs-result-item-score-box">
          {t("resultSection.score")}:{" "}
          {formatScore(result?.score ?? 0)}
        </span>
      </div>
    </div>
  );
};

// ═════════════════════════════════════════════════════════════
// ResultSection
// ═════════════════════════════════════════════════════════════

const ResultSection: React.FC<ResultSectionProps> = ({ results }) => {
  const { t } = useTranslation();
  const [activeTab, setActiveTab] = useState<ResultTab>("all");
  const [previewResult, setPreviewResult] = useState<SearchResult | null>(
    null
  );

  const safeResults = Array.isArray(results) ? results : [];

  const filteredResults = useMemo(() => {
    const filtered =
      activeTab === "all"
        ? safeResults
        : safeResults.filter((r) => r?.meta?.type === activeTab);
    return [...filtered].sort((a, b) => (b?.score ?? 0) - (a?.score ?? 0));
  }, [safeResults, activeTab]);

  return (
    <div className="cs-result-card">
      <div className="cs-result-header">
        <span className="cs-result-title">{t("resultSection.title")}</span>
      </div>
      <div className="cs-result-subtitle">{t("resultSection.subtitle")}</div>
      <div className="cs-result-tabs">
        <button
          className={`cs-result-tab ${activeTab === "all" ? "cs-result-tab--active" : ""}`}
          onClick={() => setActiveTab("all")}
        >
          {t("resultSection.all")}
        </button>
        <button
          className={`cs-result-tab ${activeTab === "document" ? "cs-result-tab--active" : ""}`}
          onClick={() => setActiveTab("document")}
        >
          {t("resultSection.documents")}
        </button>
        <button
          className={`cs-result-tab ${activeTab === "image" ? "cs-result-tab--active" : ""}`}
          onClick={() => setActiveTab("image")}
        >
          {t("resultSection.images")}
        </button>
        <button
          className={`cs-result-tab ${activeTab === "video" ? "cs-result-tab--active" : ""}`}
          onClick={() => setActiveTab("video")}
        >
          {t("resultSection.videos")}
        </button>
      </div>

      <div className="cs-result-grid">
        {filteredResults.length === 0 ? (
          <div className="cs-result-empty">
            <img
              src={searchIcon}
              alt="search"
              className="cs-result-empty-icon"
              width="48"
              height="48"
            />
            <span className="cs-result-empty-title">
              {t("resultSection.noResults")}
            </span>
            <span className="cs-result-empty-hint">
              {t("resultSection.noResultsHint")}
            </span>
          </div>
        ) : (
          filteredResults.map((result, index) => (
            <ResultCard
              key={result?.id || index}
              result={result}
              onPreview={() => setPreviewResult(result)}
            />
          ))
        )}
      </div>

      {/* Preview Modal */}
      {previewResult && (
        <PreviewModal
          result={previewResult}
          onClose={() => setPreviewResult(null)}
        />
      )}
    </div>
  );
};

export default ResultSection;
