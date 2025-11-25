import React, { useState } from 'react';
import {
  GitBranch, Link2, Upload, X,
  FolderTree, Network, BarChart2, Zap,
  Moon, Sun,
} from 'lucide-react';

const Toggle = ({ checked, onChange }) => (
  <label className="toggle">
    <input type="checkbox" checked={checked} onChange={e => onChange(e.target.checked)} />
    <span className="toggle-track" />
  </label>
);

const Sidebar = ({
  onRepoLoad,
  onRepoClear,
  activeRepoName,
  isProcessing,
  enhanceEnabled,
  toggleEnhance,
  graphEnabled,
  toggleGraph,
  treeEnabled,
  toggleTree,
  hitsEnabled,
  toggleHits,
  lightMode,
  toggleLightMode,
}) => {
  const [repoInput, setRepoInput] = useState('');

  const handleLoad = () => { if (repoInput.trim()) onRepoLoad(repoInput.trim()); };
  const handleKey  = (e) => { if (e.key === 'Enter') handleLoad(); };

  return (
    <div className="sidebar">
      {/* Logo */}
      <div className="sidebar-logo">
        <div className="sidebar-logo-icon">
          <GitBranch size={18} />
        </div>
        <h1>ChatGIT</h1>
      </div>

      {/* Repo input */}
      <div className="sidebar-section">
        <span className="sidebar-label">GitHub Repository</span>
        <div className="sidebar-input-wrap">
          <span className="sidebar-input-icon"><Link2 size={13} /></span>
          <input
            type="text"
            placeholder="https://github.com/owner/repo"
            value={repoInput}
            onChange={e => setRepoInput(e.target.value)}
            onKeyDown={handleKey}
            disabled={isProcessing}
          />
        </div>
        <button
          className="btn-primary"
          onClick={handleLoad}
          disabled={isProcessing || !repoInput.trim()}
        >
          {isProcessing
            ? <><span className="loading-ring" style={{ width: 14, height: 14, borderWidth: 2 }} /> Analyzing…</>
            : <><Upload size={14} /> Load Repository</>}
        </button>
      </div>

      {/* Active repo */}
      {activeRepoName && (
        <div className="sidebar-section">
          <span className="sidebar-label">Active Repository</span>
          <div className="repo-badge">
            <div className="repo-badge-name">
              <span className="repo-badge-dot" />
              {activeRepoName}
            </div>
          </div>
          <button className="btn-ghost" onClick={onRepoClear}>
            <X size={14} /> Clear Repository
          </button>
        </div>
      )}

      {/* View toggles */}
      <div className="sidebar-section">
        <span className="sidebar-label">Views</span>
        <div className="toggle-row">
          <span className="toggle-label"><FolderTree size={14} /> File Tree (AST)</span>
          <Toggle checked={treeEnabled} onChange={toggleTree} />
        </div>
        <div className="toggle-row">
          <span className="toggle-label"><Network size={14} /> Call Graph</span>
          <Toggle checked={graphEnabled} onChange={toggleGraph} />
        </div>
        <div className="toggle-row">
          <span className="toggle-label"><BarChart2 size={14} /> HITS Analysis</span>
          <Toggle checked={hitsEnabled} onChange={toggleHits} />
        </div>
      </div>

      {/* Feature toggles */}
      <div className="sidebar-section">
        <span className="sidebar-label">Features</span>
        <div className="toggle-row">
          <span className="toggle-label"><Zap size={14} /> Snippet Enhancement</span>
          <Toggle checked={enhanceEnabled} onChange={toggleEnhance} />
        </div>
      </div>

      {/* Footer */}
      <div className="sidebar-footer">
        <button className="theme-btn" onClick={toggleLightMode}>
          <span className="theme-btn-icon">
            {lightMode ? <Moon size={15} /> : <Sun size={15} />}
          </span>
          {lightMode ? 'Dark Mode' : 'Light Mode'}
        </button>
        <div className="sidebar-version">
          <div className="sidebar-version-name">ChatGIT v2.0</div>
          Multi-turn conversational repo intelligence with session memory, intent routing &amp; call-graph augmentation.
        </div>
      </div>
    </div>
  );
};

export default Sidebar;
