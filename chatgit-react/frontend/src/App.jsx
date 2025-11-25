import React, { useState, useEffect } from 'react';
import axios from 'axios';
import { API_BASE_URL } from './config';
import './App.css';
import Sidebar from './components/Sidebar';
import Dashboard from './components/Dashboard';
import Chat from './components/Chat';
import CallGraph from './components/CallGraph';
import StructureExplorer from './components/StructureExplorer';
import { GitBranch } from 'lucide-react';

const LOADING_STEPS = [
  { id: 'clone',   label: 'Cloning repository' },
  { id: 'ast',     label: 'Parsing AST & call graph' },
  { id: 'embed',   label: 'Embedding code chunks' },
  { id: 'index',   label: 'Building vector index' },
  { id: 'analyze', label: 'Running PageRank + HITS analysis' },
];

function App() {
  const [activeRepo, setActiveRepo]         = useState(null);
  const [repoMetrics, setRepoMetrics]       = useState(null);
  const [chatLog, setChatLog]               = useState([]);
  const [loading, setLoading]               = useState(false);
  const [loadStep, setLoadStep]             = useState(0);

  const [enhanceEnabled, setEnhance]        = useState(true);
  const [graphEnabled,   setGraph]          = useState(false);
  const [treeEnabled,    setTree]           = useState(false);
  const [hitsEnabled,    setHits]           = useState(false);
  const [lightMode,      setLightMode]      = useState(false);

  // Apply theme class to body
  useEffect(() => {
    document.body.classList.toggle('light', lightMode);
  }, [lightMode]);

  // Cycle through loading steps for visual feedback
  useEffect(() => {
    if (!loading) { setLoadStep(0); return; }
    const t = setInterval(() =>
      setLoadStep(s => (s + 1) % LOADING_STEPS.length), 2200);
    return () => clearInterval(t);
  }, [loading]);

  useEffect(() => {
    const restore = async () => {
      try {
        const r = await axios.get(`${API_BASE_URL}/api/current_repo`);
        if (r.data.repo_name) {
          setActiveRepo(r.data.repo_name);
          const [m, h] = await Promise.all([
            axios.get(`${API_BASE_URL}/api/stats`),
            axios.get(`${API_BASE_URL}/api/chat/history`),
          ]);
          setRepoMetrics(m.data);
          setChatLog(h.data || []);
        }
      } catch { /* silent */ }
    };
    restore();
  }, []);

  const loadRepo = async (url) => {
    setLoading(true);
    try {
      const r = await axios.post(`${API_BASE_URL}/api/load_repo`, { github_url: url });
      if (r.data.status === 'success') {
        setActiveRepo(r.data.repo_name);
        const m = await axios.get(`${API_BASE_URL}/api/stats`);
        setRepoMetrics(m.data);
        setChatLog([]);
      }
    } catch (err) {
      alert('Failed to load repo: ' + (err.response?.data?.detail || err.message));
    } finally {
      setLoading(false);
    }
  };

  const clearRepo = async () => {
    try {
      await axios.post(`${API_BASE_URL}/api/clear_repo`);
      setActiveRepo(null);
      setRepoMetrics(null);
      setChatLog([]);
    } catch { /* silent */ }
  };

  return (
    <div className="app-container">
      {/* Loading overlay */}
      {loading && (
        <div className="loading-overlay">
          <div className="loading-ring" />
          <h2>Analyzing Repository…</h2>
          <p>Cloning, parsing AST, embedding chunks, building vector index and running graph analysis.</p>
          <div className="loading-steps">
            {LOADING_STEPS.map((step, i) => (
              <div
                key={step.id}
                className={`loading-step ${i === loadStep ? 'active' : i < loadStep ? 'done' : ''}`}
              >
                <span className="loading-step-dot" />
                {step.label}
              </div>
            ))}
          </div>
        </div>
      )}

      <Sidebar
        onRepoLoad={loadRepo}
        onRepoClear={clearRepo}
        activeRepoName={activeRepo}
        isProcessing={loading}
        enhanceEnabled={enhanceEnabled}
        toggleEnhance={setEnhance}
        graphEnabled={graphEnabled}
        toggleGraph={setGraph}
        treeEnabled={treeEnabled}
        toggleTree={setTree}
        hitsEnabled={hitsEnabled}
        toggleHits={setHits}
        lightMode={lightMode}
        toggleLightMode={() => setLightMode(m => !m)}
      />

      <div className="main-content">
        {activeRepo ? (
          <>
            <Dashboard metrics={repoMetrics} showHits={hitsEnabled} />
            {treeEnabled && <StructureExplorer />}
            {graphEnabled && <CallGraph />}
            <Chat chatLog={chatLog} codeEnhancement={enhanceEnabled} />
          </>
        ) : (
          <div className="welcome-screen">
            <div className="welcome-orb"><GitBranch size={38} /></div>
            <h1>ChatGIT</h1>
            <p>
              Paste a GitHub URL in the sidebar to load a repository.
              ChatGIT will clone it, build a semantic index, and let you
              have a multi-turn conversation about the code.
            </p>
            <div className="welcome-features">
              {[
                'Session Memory',
                'Intent Routing',
                'Call-Graph Augmentation',
                'HITS Analysis',
                'Git Volatility',
                'PageRank',
                'Cross-Encoder Reranking',
              ].map(f => <span key={f} className="feature-chip">{f}</span>)}
            </div>
          </div>
        )}
      </div>
    </div>
  );
}

export default App;
