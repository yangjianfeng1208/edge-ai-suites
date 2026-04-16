import React, { useEffect } from "react";
import "../../assets/css/LeftPanel.css";
import UploadSection from "./UploadSection";
import SearchSection from "./SearchSection";
import { csCheckHasData } from "../../services/api";
import { useAppDispatch, useAppSelector } from "../../redux/hooks";
import { setCsHasUploads, setCsUploadsComplete, setCsDbHasData } from "../../redux/slices/uiSlice";

const ContentSearchPanel: React.FC = () => {
  const dispatch = useAppDispatch();
  const csHasUploads = useAppSelector((s) => s.ui.csHasUploads);

  // On mount, check if the backend already has indexed data (e.g. from a previous session)
  useEffect(() => {
    if (csHasUploads) return; // already known from current session
    csCheckHasData().then((hasData) => {
      if (hasData) {
        dispatch(setCsHasUploads(true));
        dispatch(setCsUploadsComplete(true));
        dispatch(setCsDbHasData(true));
      }
    });
  }, []);

  return (
    <div className="cs-panel">
      <UploadSection />
      <SearchSection />
    </div>
  );
};

export default ContentSearchPanel;
