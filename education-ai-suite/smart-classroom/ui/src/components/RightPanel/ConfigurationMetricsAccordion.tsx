import React, { useEffect, useState } from "react";
import Accordion from "../common/Accordion";
import "../../assets/css/RightPanel.css";
import { useTranslation } from "react-i18next";
import { useAppSelector } from "../../redux/hooks";
import { getConfigurationMetrics, getPlatformInfo, csGetConfig } from "../../services/api";

interface ConfigurationMetricsAccordionProps {
  activeScreen: 'main' | 'content-search';
}

const ConfigurationMetricsAccordion: React.FC<ConfigurationMetricsAccordionProps> = ({ activeScreen }) => {
  const { t } = useTranslation();
  const sessionId = useAppSelector((state) => state.ui.sessionId);
  const summaryDone = useAppSelector(
    (state) => !state.ui.aiProcessing && state.ui.summaryEnabled && !state.ui.summaryLoading
  );

  const [platformData, setPlatformData] = useState<any>(null);
  const [performanceData, setPerformanceData] = useState<any>(null);
  const [csConfigData, setCsConfigData] = useState<Record<string, string> | null>(null);

  useEffect(() => {
    if (!platformData) {
      (async () => {
        try {
          const platformResp = await getPlatformInfo();
          setPlatformData(platformResp);
        } catch (err) {
          console.error("Failed to fetch platform info:", err);
        }
      })();
    }
  }, [platformData]);

  useEffect(() => {
    if (activeScreen === 'content-search' && !csConfigData) {
      (async () => {
        try {
          const data = await csGetConfig();
          setCsConfigData(data);
        } catch (err) {
          console.error("Failed to fetch content search config:", err);
        }
      })();
    }
  }, [activeScreen, csConfigData]);

  useEffect(() => {
    setPerformanceData(null);
    if (summaryDone && sessionId) {
      (async () => {
        try {
          const configResp = await getConfigurationMetrics(sessionId);
          setPerformanceData(configResp.performance);
        } catch (err) {
          console.error("Failed to fetch performance metrics:", err);
        }
      })();
    }
  }, [summaryDone, sessionId]);

  return (
    <Accordion title={t("accordion.configuration")}>
      <div className="accordion-subtitle">
        {t("accordion.subtitle_configuration")}
      </div>

      <div className="configuration-metrics two-column">
        {/* Platform configuration */}
        <div className="platform-configuration">
          <h3>{t("accordion.platformConfiguration") || "Platform Configuration"}</h3>
          <p><strong>{t("accordion.processor") || "Processor"}:</strong> {platformData?.Processor || "-"}</p>
          <p><strong>{t("accordion.npu") || "NPU"}:</strong> {platformData?.NPU || "-"}</p>
          <p><strong>{t("accordion.igpu") || "iGPU"}:</strong> {platformData?.iGPU || "-"}</p>
          <p><strong>{t("accordion.memory") || "Memory"}:</strong> {platformData?.Memory || "-"}</p>
          <p><strong>{t("accordion.storage") || "Storage"}:</strong> {platformData?.Storage || "-"}</p>
        </div>

        {/* Software configuration */}
        <div className="software-performance">
          <h3>{t("accordion.softwareConfiguration") || "Software Configuration"}</h3>
          {activeScreen === 'main' ? (
            <>
              <p><strong>{t("accordion.llm") || "LLM"}:</strong> {platformData?.summarizer_model || "-"}</p>
              <p><strong>{t("accordion.asr") || "ASR"}:</strong> {platformData?.asr_model || "-"}</p>
            </>
          ) : (
            <>
              <p><strong>{t("accordion.vlm") || "VLM"}:</strong> {csConfigData?.vlm_model || "-"}</p>
              <p><strong>{t("accordion.visualEmbedding") || "Visual Embedding"}:</strong> {csConfigData?.visual_embedding_model || "-"}</p>
              <p><strong>{t("accordion.docEmbedding") || "Doc Embedding"}:</strong> {csConfigData?.doc_embedding_model || "-"}</p>
              <p><strong>{t("accordion.reranker") || "Reranker"}:</strong> {csConfigData?.reranker_model || "-"}</p>
              <p><strong>{t("accordion.vectorDb") || "Vector DB"}:</strong> {csConfigData?.vector_db || "-"}</p>
            </>
          )}

          {/* Performance metrics — only for main mode */}
          {activeScreen === 'main' && (
            <>
              <h3>{t("accordion.performanceMetrics") || "Performance Metrics"}</h3>
              <p><strong>{t("accordion.ttft") || "TTFT"}:</strong> {performanceData?.ttft || "-"}</p>
              <p><strong>{t("accordion.tps") || "Tokens Per Second"}:</strong> {performanceData?.tps || "-"}</p>
              <p><strong>{t("accordion.totalTokensProcessed") || "Total tokens processed"}:</strong> {performanceData?.total_tokens || "-"}</p>
              <p><strong>{t("accordion.totalTimeTaken") || "Total Time Taken"}:</strong> {performanceData?.end_to_end_time || "-"}</p>
            </>
          )}
        </div>
      </div>
    </Accordion>
  );
};

export default ConfigurationMetricsAccordion;
