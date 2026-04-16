import React, { useRef, useState, useCallback, useEffect } from "react";
import { useTranslation } from "react-i18next";
import "../../assets/css/SearchSection.css";
import { csSearch, csFetchTags } from "../../services/api";
import ResultSection, { type CsSearchResult } from "./ResultSection";
import warningIcon from "../../assets/images/warning-info.svg";
import cameraIcon from "../../assets/images/camera-icon.svg";
import noSearchIcon from "../../assets/images/no-search-icon.svg";
import { useAppDispatch, useAppSelector } from "../../redux/hooks";
import { addCsAvailableLabels } from "../../redux/slices/uiSlice";

type SearchTab = "text" | "image";
type SearchType = "document" | "image" | "video";

const MAX_QUERY_LENGTH = 100;
const DEFAULT_MAX_RESULTS = 5;

const ALLOWED_IMAGE_EXTENSIONS = new Set([".png", ".jpg", ".jpeg"]);

function isAllowedImage(filename: string): boolean {
  const ext = filename.slice(filename.lastIndexOf(".")).toLowerCase();
  return ALLOWED_IMAGE_EXTENSIONS.has(ext);
}

function fileToBase64(file: File): Promise<string> {
  return new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.onload = () => {
      const result = reader.result as string;
      const base64 = result.split(",")[1];
      resolve(base64);
    };
    reader.onerror = reject;
    reader.readAsDataURL(file);
  });
}

const SearchSection: React.FC = () => {
  const { t } = useTranslation();
  const csUploadsComplete = useAppSelector((s) => s.ui.csUploadsComplete);
  const csHasUploads = useAppSelector((s) => s.ui.csHasUploads);
  const csDbHasData = useAppSelector((s) => s.ui.csDbHasData);
  const csProcessing = useAppSelector((s) => s.ui.csProcessing);
  const csAvailableLabels = useAppSelector((s) => s.ui.csAvailableLabels);
  const dispatch = useAppDispatch();
  const imageInputRef = useRef<HTMLInputElement>(null);
  const filterBoxRef = useRef<HTMLDivElement>(null);

  // Fetch persisted tags from backend on mount and when uploads complete
  useEffect(() => {
    if (csDbHasData || csUploadsComplete) {
      csFetchTags().then((tags) => {
        if (tags.length > 0) dispatch(addCsAvailableLabels(tags));
      });
    }
  }, [csDbHasData, csUploadsComplete, dispatch]);

  const [activeTab, setActiveTab] = useState<SearchTab>("text");

  const [query, setQuery] = useState("");

  const [imageFile, setImageFile] = useState<File | null>(null);
  const [imagePreview, setImagePreview] = useState<string | null>(null);
  const [isDragOver, setIsDragOver] = useState(false);

  const [selectedTypes, setSelectedTypes] = useState<Set<SearchType>>(
    new Set(["document", "image", "video"])
  );

  const [selectedLabels, setSelectedLabels] = useState<string[]>([]);
  const [isLabelDropdownOpen, setIsLabelDropdownOpen] = useState(false);

  // Reset search when all uploads are cleared
  useEffect(() => {
    if (!csHasUploads) {
      setQuery("");
      setImageFile(null);
      if (imagePreview) {
        URL.revokeObjectURL(imagePreview);
      }
      setImagePreview(null);
      setSelectedTypes(new Set(["document", "image", "video"]));
      setSelectedLabels([]);
      setIsLabelDropdownOpen(false);
      setMaxResults(DEFAULT_MAX_RESULTS);
      setSearchResults([]);
      setShowResults(false);
      setHasSearched(false);
      setActiveTab("text");
    }
  }, [csHasUploads]);

  const [maxResults, setMaxResults] = useState<number>(DEFAULT_MAX_RESULTS);

  const [isSearching, setIsSearching] = useState(false);
  const [searchResults, setSearchResults] = useState<CsSearchResult[]>([]);
  const [showResults, setShowResults] = useState(false);
  const [hasSearched, setHasSearched] = useState(false);

  const hasValidInput = activeTab === "text" ? query.trim().length > 0 : imageFile !== null;
  const hasSelectedType = selectedTypes.size > 0;
  const canSearch = hasValidInput && hasSelectedType && !isSearching;

  const handleTabChange = useCallback((tab: SearchTab) => {
    setActiveTab(tab);
    if (tab === "image") {
      setSelectedTypes((prev) => {
        const next = new Set(prev);
        next.delete("document");
        if (next.size === 0) {
          next.add("image");
          next.add("video");
        }
        return next;
      });
    } else {
      setSelectedTypes(new Set(["document", "image", "video"]));
    }
  }, []);

  const toggleType = useCallback((type: SearchType) => {
    setSelectedTypes((prev) => {
      const next = new Set(prev);
      if (next.has(type)) {
        next.delete(type);
      } else {
        next.add(type);
      }
      return next;
    });
  }, []);

  const toggleLabel = useCallback((label: string) => {
    setSelectedLabels((prev) =>
      prev.includes(label) ? prev.filter((l) => l !== label) : [...prev, label]
    );
  }, []);

  // Close dropdown on click outside
  useEffect(() => {
    const handleClickOutside = (event: MouseEvent) => {
      if (filterBoxRef.current && !filterBoxRef.current.contains(event.target as Node)) {
        setIsLabelDropdownOpen(false);
      }
    };
    if (isLabelDropdownOpen) {
      document.addEventListener("mousedown", handleClickOutside);
    }
    return () => document.removeEventListener("mousedown", handleClickOutside);
  }, [isLabelDropdownOpen]);

  const removeLabel = useCallback((label: string) => {
    setSelectedLabels((prev) => prev.filter((l) => l !== label));
  }, []);

  const handleImageDragOver = (e: React.DragEvent) => {
    e.preventDefault();
    setIsDragOver(true);
  };

  const handleImageDragLeave = () => setIsDragOver(false);

  const handleImageDrop = (e: React.DragEvent) => {
    e.preventDefault();
    setIsDragOver(false);
    const files = Array.from(e.dataTransfer.files).filter((f) => isAllowedImage(f.name));
    if (files.length > 0) {
      processImageFile(files[0]);
    }
  };

  const handleImageBrowse = () => imageInputRef.current?.click();

  const handleImageChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const files = Array.from(e.target.files ?? []).filter((f) => isAllowedImage(f.name));
    if (files.length > 0) {
      processImageFile(files[0]);
    }
    if (imageInputRef.current) imageInputRef.current.value = "";
  };

  const processImageFile = (file: File) => {
    setImageFile(file);
    const url = URL.createObjectURL(file);
    setImagePreview(url);
  };

  const clearImage = () => {
    setImageFile(null);
    if (imagePreview) {
      URL.revokeObjectURL(imagePreview);
    }
    setImagePreview(null);
  };

  // Search handler
  const handleSearch = async () => {
    if (!canSearch) return;

    setIsSearching(true);
    setHasSearched(true);

    try {
      const filter: Record<string, string[]> = {};
      if (selectedTypes.size > 0) {
        filter.type = Array.from(selectedTypes);
      }
      if (selectedLabels.length > 0) {
        filter.tags = selectedLabels;
      }
      let results: CsSearchResult[];
      if (activeTab === "text") {
        results = await csSearch({
          query: query.trim(),
          max_num_results: maxResults,
          filter: Object.keys(filter).length > 0 ? filter : undefined,
        });
      } else {
        const base64 = await fileToBase64(imageFile!);
        results = await csSearch({
          image_base64: base64,
          max_num_results: maxResults,
          filter: Object.keys(filter).length > 0 ? filter : undefined,
        });
      }
      setSearchResults(results);
      setShowResults(true);
    } catch (error) {
      console.error("Search failed:", error);
      setSearchResults([]);
      setShowResults(true);
    } finally {
      setIsSearching(false);
    }
  };

  // Reset handler
  const handleReset = () => {
    setQuery("");
    clearImage();
    setSelectedTypes(new Set(["document", "image", "video"]));
    setSelectedLabels([]);
    setIsLabelDropdownOpen(false);
    setMaxResults(DEFAULT_MAX_RESULTS);
    setSearchResults([]);
    setShowResults(false);
    setHasSearched(false);
    setActiveTab("text");
  };

  return (
    <>
      <div className="cs-search-card">
        {/* Header */}
        <div className="cs-search-header">
          <span className="cs-search-title">{t("searchSection.title")}</span>
        </div>

        {!csDbHasData && !csUploadsComplete ? (
          /* No searchable data yet */
          <div className="cs-search-disabled">
            <img
              src={noSearchIcon}
              alt="search unavailable"
              className="cs-search-disabled-icon"
            />
            <span className="cs-search-disabled-title">{t("searchSection.searchNotAvailable")}</span>
            <span className="cs-search-disabled-hint">
              {csHasUploads ? t("searchSection.filesStillUploading") : t("searchSection.uploadFilesToEnable")}
            </span>
          </div>
        ) : (
          /* Search form — DB has data or at least one upload completed */
          <>
            {csProcessing && (
              <div className="cs-search-processing-banner">
                <span className="cs-search-processing-icon">ℹ</span>
                <span>{t("searchSection.processingBanner")}</span>
              </div>
            )}
            <div className="cs-search-tabs">
              <button
                className={`cs-search-tab ${activeTab === "text" ? "cs-search-tab--active" : ""}`}
                onClick={() => handleTabChange("text")}
              >
                {t("searchSection.textSearch")}
              </button>
              <button
                className={`cs-search-tab ${activeTab === "image" ? "cs-search-tab--active" : ""}`}
                onClick={() => handleTabChange("image")}
              >
                {t("searchSection.imageSearch")}
              </button>
            </div>

            {activeTab === "text" && (
              <div className="cs-search-content">
                <div className="cs-search-label-row">
                  <span className="cs-search-label">{t("searchSection.yourQuestion")}</span>
                  <span className="cs-search-char-count">
                    {query.length}/{MAX_QUERY_LENGTH}
                  </span>
                </div>
                <textarea
                  className="cs-search-textarea"
                  placeholder={t("searchSection.placeholder")}
                  value={query}
                  onChange={(e) => {
                    if (e.target.value.length <= MAX_QUERY_LENGTH) {
                      setQuery(e.target.value);
                    }
                  }}
                  maxLength={MAX_QUERY_LENGTH}
                />
              </div>
            )}

            {/* Image Search Content */}
            {activeTab === "image" && (
              <div className="cs-search-content">
                {!imageFile ? (
                  <div
                    className={`cs-search-dropzone ${isDragOver ? "cs-search-dropzone--active" : ""}`}
                    onDragOver={handleImageDragOver}
                    onDragLeave={handleImageDragLeave}
                    onDrop={handleImageDrop}
                    onClick={handleImageBrowse}
                  >
                    <img 
                      src={cameraIcon} 
                      alt="camera" 
                      className="cs-search-dropzone-camera"
                      width="56" 
                      height="48" 
                    />
                    <p className="cs-search-dropzone-text">
                      {t("searchSection.dragDropImage")}
                    </p>
                    <p className="cs-search-dropzone-hint">{t("searchSection.orClickBrowse")}</p>
                  </div>
                ) : (
                  <div className="cs-search-image-preview">
                    <img src={imagePreview!} alt="Search preview" />
                    <button className="cs-search-image-clear" onClick={clearImage}>
                      ✕
                    </button>
                  </div>
                )}
                <input
                  ref={imageInputRef}
                  type="file"
                  accept=".png,.jpg,.jpeg"
                  style={{ display: "none" }}
                  onChange={handleImageChange}
                />
              </div>
            )}

            {/* Search Type Selection */}
            <div className="cs-search-type-section">
              <div className="cs-search-type-label">{t("searchSection.selectSearchType")}</div>
              <div className="cs-search-type-options">
                {activeTab === "text" && (
                  <label className="cs-search-type-option">
                    <input
                      type="checkbox"
                      checked={selectedTypes.has("document")}
                      onChange={() => toggleType("document")}
                    />
                    <span>{t("searchSection.documents")}</span>
                  </label>
                )}
                <label className="cs-search-type-option">
                  <input
                    type="checkbox"
                    checked={selectedTypes.has("image")}
                    onChange={() => toggleType("image")}
                  />
                  <span>{t("searchSection.images")}</span>
                </label>
                <label className="cs-search-type-option">
                  <input
                    type="checkbox"
                    checked={selectedTypes.has("video")}
                    onChange={() => toggleType("video")}
                  />
                  <span>{t("searchSection.videos")}</span>
                </label>
              </div>
              {!hasSelectedType && (
                <div className="cs-search-warning">
                  <span className="cs-search-warning-icon">
                    <img src={warningIcon} alt="warning" width="16" height="16" />
                  </span>
                  <span>{t("searchSection.selectAtLeastOneType")}</span>
                </div>
              )}
            </div>

            {/* Section Divider */}
            <div className="cs-search-divider" />

            {/* Filter by Label */}
            {csAvailableLabels.length > 0 && (
              <div className={`cs-search-filter-section ${!hasSelectedType || !hasValidInput ? "cs-search-filter-disabled" : ""}`}>
                <div className="cs-search-filter-label">{t("searchSection.filterByLabel")}</div>
                <div
                  ref={filterBoxRef}
                  className="cs-search-filter-box"
                  onClick={() => hasSelectedType && hasValidInput && setIsLabelDropdownOpen(!isLabelDropdownOpen)}
                >
                  <div className="cs-search-filter-chips">
                    {selectedLabels.length === 0 && (
                      <span className="cs-search-filter-placeholder">
                        {t("searchSection.selectLabels", "Select labels...")}
                      </span>
                    )}
                    {selectedLabels.map((label) => (
                      <span key={label} className="cs-search-chip">
                        {label}
                        <button
                          className="cs-search-chip-remove"
                          onClick={(e) => {
                            e.stopPropagation();
                            removeLabel(label);
                          }}
                        >
                          ✕
                        </button>
                      </span>
                    ))}
                  </div>
                  {isLabelDropdownOpen && hasSelectedType && hasValidInput && (
                    <div className="cs-search-filter-dropdown" onClick={(e) => e.stopPropagation()}>
                      {csAvailableLabels.map((label) => (
                        <label key={label} className="cs-search-filter-dropdown-item">
                          <input
                            type="checkbox"
                            checked={selectedLabels.includes(label)}
                            onChange={() => toggleLabel(label)}
                          />
                          <span>{label}</span>
                        </label>
                      ))}
                    </div>
                  )}
                </div>
              </div>
            )}

            {/* Top Results */}
            <div className={`cs-search-results-section ${!hasSelectedType || !hasValidInput ? "cs-search-filter-disabled" : ""}`}>
              <div className="cs-search-results-row">
                <span className="cs-search-results-label">{t("searchSection.topResults")}</span>
                <input
                  type="number"
                  className="cs-search-results-input"
                  value={maxResults}
                  onChange={(e) => setMaxResults(Math.max(1, parseInt(e.target.value) || 1))}
                  min={1}
                  max={1000}
                  disabled={!hasSelectedType || !hasValidInput}
                />
              </div>
            </div>

            {/* Action Buttons */}
            <div className="cs-search-actions">
              <button
                className={`cs-search-btn cs-search-btn--primary ${isSearching ? "cs-search-btn--loading" : ""}`}
                onClick={handleSearch}
                disabled={!canSearch}
              >
                {isSearching && <span className="cs-spinner" />}
                {isSearching ? t("searchSection.searching") : t("searchSection.search")}
              </button>
              <button
                className="cs-search-btn cs-search-btn--secondary"
                onClick={handleReset}
              >
                {t("searchSection.reset")}
              </button>
            </div>
            <div className="cs-search-divider" />

            {/* Results Section */}
            {hasSearched && showResults && (
              <ResultSection results={searchResults} />
            )}
          </>
        )}
      </div>
    </>
  );
};

export default SearchSection;
