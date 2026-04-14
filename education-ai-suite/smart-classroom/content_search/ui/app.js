/**
 * Content Search - Main Application (Search & Backend Health Check Only)
 * File upload functionality is handled by app_ui_renderer_v3.js and app_file_manager.js
 */

/**
 * Backend Status Checker
 * Checks if the backend API is reachable
 */
const API_BASE_URL = "http://127.0.0.1:9011";
window.API_BASE_URL = API_BASE_URL; // Export for other scripts
const HEALTH_CHECK_INTERVAL = 60000; // Check every 60 seconds (1 minute)
const HEALTH_CHECK_TIMEOUT = 3000; // 3 second timeout for health check

let healthCheckTimer = null;
let currentBackendStatus = "checking";

function updateBackendStatusUI(status, message) {
  const statusEl = el("backend-status");
  const textEl = el("backend-status-text");
  const retryBtn = el("backend-status-retry");

  // Remove all status classes
  statusEl.classList.remove("backend-status--online", "backend-status--offline", "backend-status--checking");

  // Add current status class
  statusEl.classList.add(`backend-status--${status}`);

  // Update text
  textEl.textContent = message;

  // Always show retry button for manual check
  if (retryBtn) {
    retryBtn.hidden = false;
  }

  currentBackendStatus = status;
}

async function checkBackendHealth() {
  try {
    updateBackendStatusUI("checking", "Checking...");

    const controller = new AbortController();
    const timeoutId = setTimeout(() => controller.abort(), HEALTH_CHECK_TIMEOUT);

    // Use dedicated health check endpoint
    const response = await fetch(`${API_BASE_URL}/api/v1/system/health`, {
      method: "GET",
      signal: controller.signal,
    });

    clearTimeout(timeoutId);

    if (response.ok) {
      const data = await response.json();

      // Check if backend and database are healthy
      if (data.status === "ok") {
        const dbStatus = data.services?.database;
        if (dbStatus === "healthy") {
          updateBackendStatusUI("online", "Online");
          return true;
        } else {
          updateBackendStatusUI("offline", "DB Error");
          console.warn("Database unhealthy:", dbStatus);
          return false;
        }
      } else {
        updateBackendStatusUI("offline", "Unhealthy");
        return false;
      }
    } else {
      updateBackendStatusUI("offline", `Error ${response.status}`);
      return false;
    }
  } catch (error) {
    if (error.name === "AbortError") {
      updateBackendStatusUI("offline", "Timeout");
    } else {
      updateBackendStatusUI("offline", "Offline");
    }
    return false;
  }
}

function startHealthCheck() {
  // Initial check
  checkBackendHealth();

  // Setup periodic check
  if (healthCheckTimer) {
    clearInterval(healthCheckTimer);
  }
  healthCheckTimer = setInterval(checkBackendHealth, HEALTH_CHECK_INTERVAL);
}

function stopHealthCheck() {
  if (healthCheckTimer) {
    clearInterval(healthCheckTimer);
    healthCheckTimer = null;
  }
}

/**
 * Business error codes mapping
 */
const ERROR_CODES = {
  20000: "SUCCESS",
  40000: "BAD_REQUEST",
  40001: "AUTH_FAILED",
  40901: "FILE_ALREADY_EXISTS",
  50001: "FILE_TYPE_ERROR",
  50002: "TASK_NOT_FOUND",
  50003: "PROCESS_FAILED",
};

/**
 * Get user-friendly error message based on business code
 */
function getErrorMessage(code, defaultMessage) {
  const errorMessages = {
    40000: "Bad request. Please check your input and try again.",
    40001: "Authentication failed. Invalid username or password.",
    40901: "File already exists. This file has been uploaded before.",
    50001: "Unsupported file format. Please check the file type.",
    50002: "Task not found. The task may have expired or been deleted.",
    50003: "Processing failed. An internal error occurred.",
  };

  return errorMessages[code] || defaultMessage || "An unknown error occurred.";
}

/**
 * Parse API response and throw error if business code indicates failure
 */
function parseApiResponse(data, defaultErrorMsg = "Operation failed") {
  if (!data) {
    throw new Error(defaultErrorMsg);
  }

  const code = data.code;

  if (code === 20000) {
    return data;
  }

  // Non-success code
  const errorType = ERROR_CODES[code] || "UNKNOWN_ERROR";
  const userMessage = getErrorMessage(code, data.message);
  const error = new Error(userMessage);
  error.code = code;
  error.errorType = errorType;
  error.originalMessage = data.message;

  throw error;
}

function el(id) {
  return document.getElementById(id);
}

function setStatus(target, message) {
  target.textContent = message || "";
}

function formatTimestamp(totalSeconds) {
  const h = Math.floor(totalSeconds / 3600);
  const m = Math.floor((totalSeconds % 3600) / 60);
  const s = totalSeconds % 60;
  if (h > 0) return `${h}:${String(m).padStart(2, "0")}:${String(s).padStart(2, "0")}`;
  return `${String(m).padStart(2, "0")}:${String(s).padStart(2, "0")}`;
}

function setupQueryImage() {
  const input = el("query-image");

  const clear = () => {
    input.value = "";
  };

  return {
    getFile: () => input.files?.[0] || null,
    clear,
  };
}

function fileExtension(name) {
  const lower = String(name || "").toLowerCase();
  if (!lower.includes(".")) return "";
  return lower.slice(lower.lastIndexOf("."));
}

/**
 * Extract file_key from file_path for download API
 * file_path format: "local://content-search/runs/xxx/raw/video/default/test.mp4"
 * file_key format: "runs/xxx/raw/video/default/test.mp4"
 */
function extractFileKey(filePath) {
  if (!filePath) return null;
  // Remove protocol and bucket name
  return filePath.replace(/^[a-z]+:\/\/[^/]+\//, '');
}

/**
 * Open preview modal
 */
function openPreviewModal(result) {
  const modal = el("preview-modal");
  const title = el("preview-modal-title");
  const body = el("preview-modal-body");

  if (!modal || !title || !body) return;

  title.textContent = result.filename || "Preview";
  body.innerHTML = '<div class="preview-loading">Loading preview...</div>';

  // Show modal
  modal.hidden = false;
  document.body.style.overflow = "hidden";

  // Get file key for download
  const fileKey = extractFileKey(result.file_path);
  if (!fileKey) {
    body.innerHTML = '<div class="preview-error">Cannot preview this file (invalid path)</div>';
    return;
  }

  // Use inline=true for preview (browser displays), without for download
  const previewUrl = `${API_BASE_URL}/api/v1/object/download?file_key=${encodeURIComponent(fileKey)}&inline=true`;
  const downloadUrl = `${API_BASE_URL}/api/v1/object/download?file_key=${encodeURIComponent(fileKey)}`;

  // Render preview based on type
  setTimeout(() => {
    try {
      let content = '';

      // Info section
      content += '<div class="preview-info">';
      content += `<div class="preview-info__row"><span class="preview-info__label">Filename:</span><span class="preview-info__value">${result.filename}</span></div>`;
      content += `<div class="preview-info__row"><span class="preview-info__label">Type:</span><span class="preview-info__value">${result.type}</span></div>`;
      content += `<div class="preview-info__row"><span class="preview-info__label">Score:</span><span class="preview-info__value">${result.score}%</span></div>`;

      if (result.type === 'video' && result.timestamp) {
        content += `<div class="preview-info__row"><span class="preview-info__label">Match Time:</span><span class="preview-info__value">${result.timestamp}</span></div>`;
      } else if (result.type === 'document' && result.page) {
        content += `<div class="preview-info__row"><span class="preview-info__label">Page:</span><span class="preview-info__value">${result.page}</span></div>`;
      }

      content += '</div>';

      // Preview content based on type
      if (result.type === 'video') {
        const videoSrc = previewUrl;
        const startTime = result.video_pin_second || 0;

        // Add hint if there's a specific timestamp
        if (startTime > 0) {
          content += `<div class="preview-hint">
            <svg width="16" height="16" viewBox="0 0 16 16" fill="currentColor" style="margin-right: 4px;">
              <circle cx="8" cy="8" r="7" fill="none" stroke="currentColor" stroke-width="1.5"/>
              <path d="M8 4v4l3 2" stroke="currentColor" stroke-width="1.5" fill="none" stroke-linecap="round"/>
            </svg>
            Video will start at ${result.timestamp || formatTimestamp(Math.floor(startTime))} (matched scene)
          </div>`;
        }

        // Show summary text if available
        if (result.summary) {
          content += `<div class="preview-context">
            <div class="preview-context__label">Scene Summary:</div>
            <div class="preview-context__text">${result.summary}</div>
          </div>`;
        }

        content += `<video class="preview-video" controls autoplay>
          <source src="${videoSrc}#t=${startTime}" type="video/mp4">
          Your browser does not support the video tag.
        </video>`;
      } else if (result.type === 'image') {
        content += `<img class="preview-image" src="${previewUrl}" alt="${result.filename}">`;
      } else if (result.type === 'document') {
        if (result.subtype === 'pdf') {
          // Show matched text context if available
          if (result.chunk_text) {
            content += `<div class="preview-context">
              <div class="preview-context__label">Matched Content:</div>
              <div class="preview-context__text">"${result.chunk_text}"</div>
            </div>`;
          }
          // Show PDF viewer (use previewUrl for inline display)
          content += `<div class="preview-pdf-container">
            <iframe class="preview-pdf" src="${previewUrl}" type="application/pdf"></iframe>
          </div>`;
        } else if (result.summary) {
          content += `<div class="preview-text">${result.summary}</div>`;
        } else {
          content += `<div class="preview-info">
            <p>Document preview not available. <a href="${downloadUrl}" download>Click here to download</a></p>
          </div>`;
        }
      }

      body.innerHTML = content;

      // If video, ensure we seek to the exact timestamp (multiple fallbacks)
      if (result.type === 'video' && result.video_pin_second) {
        const video = body.querySelector('video');
        if (video) {
          const targetTime = result.video_pin_second;

          // Method 1: Set on loadedmetadata (most reliable)
          video.addEventListener('loadedmetadata', () => {
            video.currentTime = targetTime;
          });

          // Method 2: Set on canplay (backup)
          video.addEventListener('canplay', () => {
            if (Math.abs(video.currentTime - targetTime) > 0.5) {
              video.currentTime = targetTime;
            }
          });

          // Method 3: Force set after a short delay (final fallback)
          setTimeout(() => {
            if (video.readyState >= 2 && Math.abs(video.currentTime - targetTime) > 0.5) {
              video.currentTime = targetTime;
            }
          }, 500);
        }
      }
    } catch (error) {
      body.innerHTML = `<div class="preview-error">Error loading preview: ${error.message}</div>`;
    }
  }, 100);
}

/**
 * Close preview modal
 */
function closePreviewModal() {
  const modal = el("preview-modal");
  const body = el("preview-modal-body");

  if (modal) {
    modal.hidden = true;
    document.body.style.overflow = "";
  }

  // Stop any playing media
  if (body) {
    const video = body.querySelector('video');
    if (video) {
      video.pause();
      video.src = "";
    }
    body.innerHTML = "";
  }
}

function renderResults(listEl, metaEl, results, metaText) {
  const resultsContainer = el("results-container");

  // Update data-empty attribute to control header visibility
  if (resultsContainer) {
    const isEmpty = !results || results.length === 0;
    resultsContainer.setAttribute("data-empty", isEmpty ? "true" : "false");
  }

  metaEl.textContent = metaText;
  listEl.innerHTML = "";
  results.forEach((r) => {
    const li = document.createElement("li");
    li.className = "result-card";

    // ── Header: filename · preview icon · score ──────────────────
    const header = document.createElement("div");
    header.className = "result-card__header";

    // Filename container with preview icon (icon on left)
    const filenameContainer = document.createElement("div");
    filenameContainer.className = "result-card__filename-container";

    // Preview icon button (moved to left)
    const previewIcon = document.createElement("button");
    previewIcon.className = "result-card__preview-icon";
    previewIcon.type = "button";
    previewIcon.title = "Preview";
    previewIcon.innerHTML = `<svg width="16" height="16" viewBox="0 0 16 16" fill="none" stroke="currentColor" stroke-width="1.5">
      <circle cx="8" cy="8" r="3"/>
      <path d="M2 8s2-4 6-4 6 4 6 4-2 4-6 4-6-4-6-4z"/>
    </svg>`;
    previewIcon.addEventListener("click", (e) => {
      e.stopPropagation();
      openPreviewModal(r);
    });

    const filename = document.createElement("h4");
    filename.className = "result-card__filename";
    filename.textContent = r.filename || r.title || "Unknown";
    filename.title = r.filename; // Show full name on hover

    filenameContainer.appendChild(previewIcon);
    filenameContainer.appendChild(filename);

    const scoreEl = document.createElement("span");
    const sv = typeof r.score === "number" ? r.score : null;
    if (sv !== null) {
      scoreEl.className = "result-card__score" +
        (sv >= 85 ? " result-card__score--high" : sv >= 60 ? " result-card__score--mid" : " result-card__score--low");
      scoreEl.textContent = `Score: ${sv}%`;
    }

    header.append(filenameContainer, scoreEl);
    li.appendChild(header);

    // ── Metadata list ─────────────────────────────
    const detailsList = document.createElement("ul");
    detailsList.className = "result-card__details";

    // Page or timestamp information
    if (r.type === "document" && r.page) {
      const pageItem = document.createElement("li");
      pageItem.textContent = r.page;
      detailsList.appendChild(pageItem);
    } else if (r.type === "video") {
      if (r.timestamp) {
        const timeItem = document.createElement("li");
        timeItem.textContent = `Time: ${r.timestamp}`;
        detailsList.appendChild(timeItem);
      } else if (r.time_range) {
        const timeItem = document.createElement("li");
        timeItem.textContent = `Time: ${r.time_range}`;
        detailsList.appendChild(timeItem);
      }
    }

    // Labels
    if (r.labels && r.labels.length) {
      const labelsItem = document.createElement("li");
      labelsItem.className = "result-card__labels-item";

      const labelText = document.createElement("span");
      labelText.textContent = "Labels: ";
      labelText.style.marginRight = "8px";
      labelsItem.appendChild(labelText);

      r.labels.forEach((lbl) => {
        const chip = document.createElement("span");
        chip.className = "tag tag--sm";
        chip.textContent = lbl;
        labelsItem.appendChild(chip);
      });

      detailsList.appendChild(labelsItem);
    }

    if (detailsList.children.length > 0) {
      li.appendChild(detailsList);
    }

    listEl.appendChild(li);
  });
}

function clampIntegerInInput(inputEl, { min, max, fallback }) {
  const raw = String(inputEl.value ?? "").trim();
  // Allow user to temporarily clear the field while typing.
  if (raw === "") return;

  const parsed = Number.parseInt(raw, 10);
  if (!Number.isFinite(parsed)) {
    inputEl.value = String(fallback);
    return;
  }

  const clamped = Math.max(min, Math.min(max, parsed));
  if (clamped !== parsed) inputEl.value = String(clamped);
}

function filterResultsByType(results, filterValue) {
  const value = String(filterValue || "all").toLowerCase();
  if (value === "all") return results;
  const allowed = new Set(["document", "image", "video"]);
  if (!allowed.has(value)) return results;
  return results.filter((r) => String(r?.type || "").toLowerCase() === value);
}

(function main() {
  // Note: File upload functionality is handled by app_ui_renderer_v3.js and app_file_manager.js
  // This file only handles search functionality and backend health check

  const searchStatus = el("search-status");
  const resultsMeta = el("results-meta");
  const resultsList = el("results-list");
  const resultsFilterEl = el("results-filter");

  /** @type {{ type?: string, title: string, meta: string }[]} */
  let lastResults = [];
  /** @type {string} */
  let lastResultsMetaText = "";

  // Global pool of all labels (populated from fileManager in app_ui_renderer.js)
  const availableLabels = window.fileManagerUI?.fileManager?.availableLabels || new Set();
  const NO_LABEL_KEY = "__no_label__";
  const searchSelectedLabels = new Set();

  // New tag input UI elements
  const labelTagsList = el("label-tags-list");
  const labelTagAddBtn = el("label-tag-add-btn");

  /**
   * Create a new editable tag
   */
  function createEditableTag(initialValue = "") {
    const tag = document.createElement("div");
    tag.className = "label-tag-item";

    const input = document.createElement("input");
    input.type = "text";
    input.className = "label-tag-item__input";
    input.placeholder = "Enter label...";
    input.value = initialValue;

    const removeBtn = document.createElement("button");
    removeBtn.type = "button";
    removeBtn.className = "label-tag-item__remove";
    removeBtn.innerHTML = `<svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
      <path d="M18 6L6 18M6 6l12 12"/>
    </svg>`;

    removeBtn.addEventListener("click", () => {
      const value = input.value.trim();
      if (value) {
        searchSelectedLabels.delete(value);
      }
      tag.remove();
    });

    // When user types, update the set
    input.addEventListener("blur", () => {
      const oldValue = input.dataset.oldValue || "";
      const newValue = input.value.trim();

      if (oldValue && oldValue !== newValue) {
        searchSelectedLabels.delete(oldValue);
      }

      if (newValue) {
        searchSelectedLabels.add(newValue);
        input.dataset.oldValue = newValue;
      } else {
        tag.remove();
      }
    });

    // Enter key to confirm
    input.addEventListener("keypress", (e) => {
      if (e.key === "Enter") {
        input.blur();
      }
    });

    tag.appendChild(input);
    tag.appendChild(removeBtn);

    return { tag, input };
  }

  /**
   * Add a new tag input
   */
  labelTagAddBtn?.addEventListener("click", () => {
    const { tag, input } = createEditableTag();
    labelTagsList.insertBefore(tag, labelTagAddBtn);
    input.focus();
  });

  /**
   * Render existing tags from searchSelectedLabels
   */
  function renderLabelTags() {
    // Clear existing tags (but keep the + button)
    const existingTags = labelTagsList.querySelectorAll(".label-tag-item");
    existingTags.forEach((t) => t.remove());

    // Re-render from set
    searchSelectedLabels.forEach((label) => {
      if (label && label !== NO_LABEL_KEY) {
        const { tag } = createEditableTag(label);
        tag.querySelector("input").dataset.oldValue = label;
        labelTagsList.insertBefore(tag, labelTagAddBtn);
      }
    });
  }

  // Old dropdown functions removed - using tag input now

  // Old dropdown render function removed

  // Old dropdown event listeners removed - using tag input now

  const queryImage = setupQueryImage();
  const queryTextEl = el("query-text");
  const queryImageEl = el("query-image");
  const queryAttachBtn = el("query-attach-btn");
  const queryImagePreviewContainer = el("query-image-preview-container");
  const queryImagePreviewImg = el("query-image-preview-img");
  const queryImagePreviewName = el("query-image-preview-name");
  const queryImageClearBtn = el("query-image-clear");
  const topKSelectEl = el("topk-select");

  // Auto-resize textarea based on content
  function autoResizeTextarea() {
    if (!queryTextEl) return;

    // Reset height to min-height to get accurate scrollHeight
    queryTextEl.style.height = 'auto';

    // Calculate new height based on content
    const newHeight = Math.max(68, queryTextEl.scrollHeight); // 68px is min-height (2 rows)
    queryTextEl.style.height = newHeight + 'px';
  }

  // Initialize textarea height and add event listeners
  if (queryTextEl) {
    autoResizeTextarea();
    queryTextEl.addEventListener('input', autoResizeTextarea);
    // Also resize on paste
    queryTextEl.addEventListener('paste', () => {
      setTimeout(autoResizeTextarea, 0);
    });
  }

  // Search type checkboxes and dropdown
  const typeDocEl = el("type-document");
  const typeImgEl = el("type-image");
  const typeVidEl = el("type-video");
  const typeDocLabel = typeDocEl?.closest("label");
  const typesDropdown = el("types-dropdown");
  const typesDropdownBtn = el("types-dropdown-btn");
  const typesDropdownMenu = el("types-dropdown-menu");
  const typesDropdownLabel = el("types-dropdown-label");

  /** @type {"text" | "image"} */
  let queryMode = "text";

  let docCheckedBeforeImageMode = true;

  // Update dropdown label based on selected types
  const updateTypesDropdownLabel = () => {
    const selected = getSelectedTypes();
    if (selected.length === 0) {
      typesDropdownLabel.textContent = "Types";
    } else if (selected.length === 3) {
      typesDropdownLabel.textContent = "All Types";
    } else {
      typesDropdownLabel.textContent = selected.length === 1 ? selected[0] : `${selected.length} types`;
    }
  };

  // Toggle dropdown menu
  typesDropdownBtn?.addEventListener("click", (e) => {
    e.stopPropagation();
    const isOpen = !typesDropdownMenu.hidden;
    typesDropdownMenu.hidden = isOpen;
    typesDropdown?.classList.toggle("is-open", !isOpen);
  });

  // Close dropdown when clicking outside
  document.addEventListener("click", (e) => {
    if (typesDropdown && !typesDropdown.contains(e.target)) {
      typesDropdownMenu.hidden = true;
      typesDropdown.classList.remove("is-open");
    }
  });

  // Update label when checkboxes change
  [typeDocEl, typeImgEl, typeVidEl].forEach(checkbox => {
    checkbox?.addEventListener("change", () => {
      updateTypesDropdownLabel();
    });
  });

  // Get selected types from checkboxes
  const getSelectedTypes = () => {
    const selected = [];
    if (typeDocEl?.checked) selected.push("document");
    if (typeImgEl?.checked) selected.push("image");
    if (typeVidEl?.checked) selected.push("video");
    return selected;
  };

  // Enforce type rules for image query mode
  const enforceTypeRulesForMode = (mode) => {
    if (mode === "image") {
      // Image query: allow only image/video (document not supported).
      docCheckedBeforeImageMode = typeDocEl?.checked || false;
      if (typeDocEl) typeDocEl.checked = false;
      if (typeDocEl) typeDocEl.disabled = true;
      typeDocLabel?.classList.add("is-disabled");
      typeDocLabel?.setAttribute("aria-disabled", "true");
    } else {
      if (typeDocEl) typeDocEl.disabled = false;
      typeDocLabel?.classList.remove("is-disabled");
      typeDocLabel?.removeAttribute("aria-disabled");
      if (typeDocEl) typeDocEl.checked = Boolean(docCheckedBeforeImageMode);
    }
    updateTypesDropdownLabel(); // Update dropdown label after changing types
  };

  const setQueryMode = (mode) => {
    queryMode = mode;
    const isImage = mode === "image";

    // Disable text input when image is selected
    queryTextEl.disabled = isImage;
    if (isImage) {
      queryTextEl.value = "";
      queryTextEl.placeholder = "Image search active...";
    } else {
      queryTextEl.placeholder = "Type your query or upload an image...";
    }

    enforceTypeRulesForMode(mode);
  };

  // Attach button click - trigger file input
  queryAttachBtn?.addEventListener("click", () => {
    queryImageEl.click();
  });

  // Image file selected
  queryImageEl.addEventListener("change", () => {
    const file = queryImage.getFile();
    if (file) {
      // Show preview
      const reader = new FileReader();
      reader.onload = (e) => {
        queryImagePreviewImg.src = e.target.result;
        queryImagePreviewName.textContent = file.name;
        queryImagePreviewContainer.hidden = false;
      };
      reader.readAsDataURL(file);

      // Switch to image mode
      setQueryMode("image");
    }
  });

  // Clear image button
  queryImageClearBtn?.addEventListener("click", () => {
    queryImage.clear();
    queryImageEl.value = "";
    queryImagePreviewContainer.hidden = true;
    setQueryMode("text");
  });

  // If user starts typing, clear image if present
  queryTextEl.addEventListener("input", () => {
    if ((queryTextEl.value || "").trim().length > 0 && queryMode === "image") {
      // User is typing, clear image
      queryImage.clear();
      queryImageEl.value = "";
      queryImagePreviewContainer.hidden = true;
      setQueryMode("text");
    }
  });

  // Top K is now a select dropdown, no validation needed

  const renderFilteredResults = () => {
    const filtered = filterResultsByType(lastResults, resultsFilterEl?.value);

    const filterLabel = String(resultsFilterEl?.value || "all").toLowerCase();
    const filterMeta = filterLabel === "all" ? "All" : filterLabel;
    const meta = lastResultsMetaText
      ? `${lastResultsMetaText} · Filter: ${filterMeta} · Showing: ${filtered.length}/${lastResults.length}`
      : "";

    renderResults(resultsList, resultsMeta, filtered, meta);
  };

  resultsFilterEl?.addEventListener("change", () => {
    renderFilteredResults();
  });

  /**
   * Perform text search via API
   */
  async function performTextSearch(query, maxResults, filter) {
    const requestBody = {
      query: query.trim(),
      max_num_results: maxResults,
    };

    if (Object.keys(filter).length > 0) {
      requestBody.filter = filter;
    }

    const response = await fetch(`${API_BASE_URL}/api/v1/object/search`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(requestBody),
    });

    if (!response.ok) {
      const errText = await response.text();
      throw new Error(errText || `HTTP ${response.status}: ${response.statusText}`);
    }

    const data = await response.json();
    parseApiResponse(data, "Search failed");

    return (data.data?.results || []).map(mapApiResultToUi);
  }

  /**
   * Perform image search via API (Base64 encoded)
   */
  async function performImageSearch(imageFile, maxResults, filter) {
    return new Promise((resolve, reject) => {
      const reader = new FileReader();

      reader.onload = async () => {
        try {
          const base64 = reader.result.split(',')[1];

          const requestBody = {
            image_base64: base64,
            max_num_results: maxResults,
          };

          if (Object.keys(filter).length > 0) {
            requestBody.filter = filter;
          }

          const response = await fetch(`${API_BASE_URL}/api/v1/object/search`, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify(requestBody),
          });

          if (!response.ok) {
            const errText = await response.text();
            throw new Error(errText || `HTTP ${response.status}: ${response.statusText}`);
          }

          const data = await response.json();
          parseApiResponse(data, "Image search failed");

          resolve((data.data?.results || []).map(mapApiResultToUi));
        } catch (error) {
          reject(error);
        }
      };

      reader.onerror = () => reject(new Error("Failed to read image file"));
      reader.readAsDataURL(imageFile);
    });
  }

  /**
   * Map API response to UI display format
   */
  function mapApiResultToUi(apiResult) {
    const meta = apiResult.meta || {};
    const type = meta.type || "document";

    // Use score directly from API response (already calculated as percentage)
    const scorePercent = Math.round(apiResult.score || 0);

    const result = {
      type: type,
      score: scorePercent,
      labels: meta.tags || [],
      // Use file_name if available, otherwise extract from file_path
      filename: meta.file_name || extractFilename(meta.file_path) || "Unknown",
      summary: meta.summary_text || meta.chunk_text || "",
      // Keep original data for preview
      file_path: meta.file_path,
      video_pin_second: meta.video_pin_second,
    };

    // Type-specific fields
    if (type === "video") {
      // Use video_pin_second for single timestamp
      if (meta.video_pin_second !== undefined) {
        result.timestamp = formatTimestamp(Math.floor(meta.video_pin_second));
      }
      // Support range if start_time and end_time exist
      if (meta.start_time !== undefined && meta.end_time !== undefined) {
        result.time_range = `${formatTimestamp(Math.floor(meta.start_time))} - ${formatTimestamp(Math.floor(meta.end_time))}`;
      }
    }

    if (type === "document") {
      // Priority: Check if it's a PDF first (has doc_filetype or doc_page_number)
      if (meta.doc_filetype === "application/pdf" || meta.doc_page_number !== undefined) {
        result.subtype = "pdf";
        // Handle page information
        if (meta.doc_page_number !== undefined) {
          result.page = `Page ${meta.doc_page_number}`;
        } else if (meta.page_range) {
          result.page = meta.page_range;
        }
        // Keep chunk_text for context
        if (meta.chunk_text) {
          result.chunk_text = meta.chunk_text;
        }
      } else if (meta.chunk_id || meta.chunk_text) {
        // Then check if it's a video summary
        result.subtype = "video_summary";
        result.chunk_id = meta.chunk_id;
      } else {
        result.subtype = "txt";
      }
    }

    if (type === "image") {
      // Add any image-specific metadata here
    }

    return result;
  }

  /**
   * Extract filename from file path
   */
  function extractFilename(filePath) {
    if (!filePath) return null;
    // Remove protocol prefix (local://, etc.)
    const path = filePath.replace(/^[a-z]+:\/\/[^/]+\//, '');
    // Get last part of path
    const parts = path.split('/');
    return parts[parts.length - 1];
  }

  el("btn-search").addEventListener("click", async () => {
    const selectedTypes = getSelectedTypes();
    const topK = Number.parseInt(topKSelectEl?.value || "4", 10);
    const textQuery = el("query-text").value || "";
    const imageFile = queryImage.getFile();

    const hasText = queryMode === "text" && textQuery.trim().length > 0;
    const hasImage = queryMode === "image" && Boolean(imageFile);

    if (!hasText && !hasImage) {
      setStatus(searchStatus, queryMode === "text" ? "Enter text before searching." : "Choose an image before searching.");
      return;
    }

    if (selectedTypes.length === 0) {
      setStatus(searchStatus, "Select at least one search type (Documents/Image/Video).");
      return;
    }

    if (queryMode === "image" && selectedTypes.includes("document")) {
      setStatus(searchStatus, "Image query supports Image and Video types only.");
      return;
    }

    const safeTopK = Number.isFinite(topK) ? Math.max(1, Math.min(10, topK)) : 4;

    // Build filter object
    const filter = {};
    if (selectedTypes.length > 0 && selectedTypes.length < 3) {
      filter.type = selectedTypes;
    }

    // Add label filter if labels are selected
    const chosenLabels = Array.from(searchSelectedLabels).filter((l) => l !== NO_LABEL_KEY);
    if (chosenLabels.length > 0 && chosenLabels.length < availableLabels.size) {
      filter.tags = chosenLabels;
    }

    setStatus(searchStatus, "Searching...");

    try {
      let results;

      if (hasImage) {
        // Image search - convert to Base64
        results = await performImageSearch(imageFile, safeTopK, filter);
      } else {
        // Text search
        results = await performTextSearch(textQuery, safeTopK, filter);
      }

      const typeLabel = selectedTypes.length === 3 ? "document, image, video" : selectedTypes.join(", ");
      const includesNoLabel = searchSelectedLabels.has(NO_LABEL_KEY);
      const labelParts = [...chosenLabels, ...(includesNoLabel ? ["(No Label)"] : [])];
      const labelNote = labelParts.length ? ` · Labels: [${labelParts.join(", ")}]` : "";
      setStatus(searchStatus, `Search completed: ${results.length} results found.`);

      lastResults = results;
      lastResultsMetaText = `Types: ${typeLabel} · TopK: ${safeTopK} · Mode: ${hasImage ? "image" : "text"}${labelNote}`;
      renderFilteredResults();
    } catch (error) {
      setStatus(searchStatus, `Search failed: ${error?.message || "Unknown error"}`);
      lastResults = [];
      lastResultsMetaText = "";
      renderFilteredResults();
    }
  });

  el("btn-reset").addEventListener("click", () => {
    searchSelectedLabels.clear();
    renderLabelTags(); // Clear all label tags

    el("query-text").value = "";
    queryImage.clear();
    queryImageEl.value = "";
    queryImagePreviewContainer.hidden = true;

    setQueryMode("text");

    if (typeDocEl) typeDocEl.checked = true;
    if (typeImgEl) typeImgEl.checked = true;
    if (typeVidEl) typeVidEl.checked = true;
    if (topKSelectEl) topKSelectEl.value = "4";
    updateTypesDropdownLabel(); // Update dropdown label

    setStatus(searchStatus, "");
    resultsMeta.textContent = "";
    resultsList.innerHTML = "";
    if (resultsFilterEl) resultsFilterEl.value = "all";
    lastResults = [];
    lastResultsMetaText = "";

    // Mark results container as empty
    const resultsContainer = el("results-container");
    if (resultsContainer) {
      resultsContainer.setAttribute("data-empty", "true");
    }

    // Reset textarea height
    autoResizeTextarea();
  });

  // Initialize UI state
  setQueryMode("text");
  renderLabelTags(); // Initialize with no tags
  updateTypesDropdownLabel(); // Initialize dropdown label

  // Setup preview modal close handlers
  const previewModalClose = el("preview-modal-close");
  const previewModalOverlay = el("preview-modal-overlay");

  if (previewModalClose) {
    previewModalClose.addEventListener("click", closePreviewModal);
  }

  if (previewModalOverlay) {
    previewModalOverlay.addEventListener("click", closePreviewModal);
  }

  // Close modal on Escape key
  document.addEventListener("keydown", (e) => {
    if (e.key === "Escape") {
      const modal = el("preview-modal");
      if (modal && !modal.hidden) {
        closePreviewModal();
      }
    }
  });
})();

// Backend status retry button - moved outside IIFE to ensure it runs after DOM loads
(function initBackendStatus() {
  const retryBtn = el("backend-status-retry");
  if (retryBtn) {
    retryBtn.addEventListener("click", () => {
      checkBackendHealth();
    });
  }

  // Start backend health check
  startHealthCheck();
})();

// Focus mode - click card headers to expand them
(function initFocusMode() {
  const mainContainer = document.querySelector('.main');
  const cards = document.querySelectorAll('.main > .card');

  if (!mainContainer || cards.length !== 2) return;

  const [leftCard, rightCard] = cards;
  const leftHeader = leftCard.querySelector('.card__header');
  const rightHeader = rightCard.querySelector('.card__header');

  if (!leftHeader || !rightHeader) return;

  let currentFocus = null; // 'left', 'right', or null

  // Add pointer cursor to headers
  leftHeader.style.cursor = 'pointer';
  rightHeader.style.cursor = 'pointer';

  leftHeader.addEventListener('click', () => {
    if (currentFocus === 'left') {
      // Click again to reset
      mainContainer.classList.remove('focus-left');
      currentFocus = null;
    } else {
      // Focus left panel
      mainContainer.classList.remove('focus-right');
      mainContainer.classList.add('focus-left');
      currentFocus = 'left';
    }
  });

  rightHeader.addEventListener('click', () => {
    if (currentFocus === 'right') {
      // Click again to reset
      mainContainer.classList.remove('focus-right');
      currentFocus = null;
    } else {
      // Focus right panel
      mainContainer.classList.remove('focus-left');
      mainContainer.classList.add('focus-right');
      currentFocus = 'right';
    }
  });
})();
