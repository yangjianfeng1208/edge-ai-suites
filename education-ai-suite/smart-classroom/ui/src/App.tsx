import React, { useEffect, useRef, useState } from 'react';
import { createPortal } from 'react-dom'; // Add this import
import TopPanel from './components/TopPanel/TopPanel';
import HeaderBar from './components/Header/Header';
import Body from './components/common/Body';
import Footer from './components/Footer/Footer';
import Modal from './components/Modals/Modal'; // Import your existing Modal
import SettingsForm from './components/Modals/SettingsForm'; // Import your existing SettingsForm
import './App.css';
import MetricsPoller from './components/common/MetricsPoller';
import { getSettings, pingBackend } from './services/api';
import { useVideoPipelineMonitor } from "../src/redux/videoMonitor";
import { useTranslation } from 'react-i18next';
  
const App: React.FC = () => {
  const { t } = useTranslation();
  const [projectName, setProjectName] = useState<string>('');
  const [isSettingsOpen, setIsSettingsOpen] = useState(false);
  const [backendStatus, setBackendStatus] = useState<'checking' | 'available' | 'unavailable'>('checking');
  const [activeScreen, setActiveScreen] = useState<'main' | 'content-search'>('main');
  useVideoPipelineMonitor();

  const mainBackendAvailable = backendStatus === 'available';

  const checkBackendHealth = async () => {
    try {
      const isHealthy = await pingBackend();

      if (isHealthy) {
        setBackendStatus('available');
        loadSettings();
        return;
      }

      setBackendStatus('unavailable');
    } catch {
      setBackendStatus('unavailable');
    }
  };

  const loadSettings = async () => {
    try {
      const settings = await getSettings();
      if (settings.projectName) setProjectName(settings.projectName);
    } catch {
      console.warn('Failed to fetch project settings');
    }
  };

  useEffect(() => {
    checkBackendHealth(); 
  }, []);

  useEffect(() => {
    if (backendStatus === 'available') return;

    const interval = setInterval(checkBackendHealth, 5000);
    return () => clearInterval(interval);
  }, [backendStatus]);

  // Auto-switch to content-search when main backend (main.py) is not running.
  // This allows using content-search independently without starting main.py.
  useEffect(() => {
    if (!mainBackendAvailable && activeScreen === 'main') {
      setActiveScreen('content-search');
    }
  }, [mainBackendAvailable]);

  // Prevent switching to main screen when its backend is unavailable
  const handleSetActiveScreen = (screen: 'main' | 'content-search') => {
    if (screen === 'main' && !mainBackendAvailable) return;
    setActiveScreen(screen);
  };

    if (backendStatus === 'checking') {
    return (
      <div className="app-loading">
        <div className="loading-content">
          <div className="spinner" />
          <h2>Checking backend status</h2>
          <p>Please wait while we connect to the backend…</p>
        </div>
      </div>
    );
  }

  return (
    <div className="app">
      {mainBackendAvailable && <MetricsPoller />}
      <TopPanel
        projectName={projectName}
        setProjectName={setProjectName}
        isSettingsOpen={isSettingsOpen}
        setIsSettingsOpen={setIsSettingsOpen}
        activeScreen={activeScreen}
        setActiveScreen={handleSetActiveScreen}
        mainBackendAvailable={mainBackendAvailable}
      />
      {mainBackendAvailable && activeScreen === 'main' && (
        <HeaderBar projectName={projectName} setProjectName={setProjectName} />
      )}
      {activeScreen === 'content-search' && (
        <div className="content-search-subheader">
          <span>{t('contentSearch.subtitle')}</span>
        </div>
      )}
      <div className="main-content">
        <Body isModalOpen={isSettingsOpen} activeScreen={activeScreen} />
      </div>
      <Footer />
      
      {/* Settings modal - only relevant when main backend is running */}
      {mainBackendAvailable && createPortal(
        <Modal
          isOpen={isSettingsOpen}
          onClose={() => setIsSettingsOpen(false)}
          showCloseIcon={true}
        >
          <SettingsForm 
            onClose={() => setIsSettingsOpen(false)}
            projectName={projectName}
            setProjectName={setProjectName}
          />
        </Modal>,
        document.body
      )}
    </div>
  );
};

export default App;