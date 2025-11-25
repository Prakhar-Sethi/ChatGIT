import React, { useState, useEffect } from 'react';
import axios from 'axios';
import { API_BASE_URL } from '../config';
import {
  Folder, FolderOpen, FileCode, FileText, File, ChevronRight, ChevronDown,
} from 'lucide-react';
import { FolderTree } from 'lucide-react';

const FileIcon = ({ name }) => {
  if (name.endsWith('.py'))                        return <FileCode size={14} className="tree-icon-py" />;
  if (name.endsWith('.js') || name.endsWith('.jsx')) return <FileCode size={14} className="tree-icon-js" />;
  if (name.endsWith('.ts') || name.endsWith('.tsx')) return <FileCode size={14} className="tree-icon-ts" />;
  if (name.endsWith('.md'))                         return <FileText size={14} className="tree-icon-md" />;
  return <File size={14} className="tree-icon-default" />;
};

const TreeNode = ({ name, data, isFile, depth }) => {
  const [expanded, setExpanded] = useState(false);
  const indent = depth * 18;

  if (isFile) {
    const functions = data.functions || [];
    const classes   = data.classes   || [];
    return (
      <div>
        <div
          className={`tree-node-row ${expanded ? 'active' : ''}`}
          style={{ paddingLeft: indent + 10 }}
          onClick={() => setExpanded(e => !e)}
        >
          <FileIcon name={name} />
          <span>{name}</span>
          {(functions.length > 0 || classes.length > 0) && (
            <span className="tree-node-counts">
              {classes.length > 0 && `${classes.length} cls`}
              {classes.length > 0 && functions.length > 0 && ' · '}
              {functions.length > 0 && `${functions.length} fn`}
            </span>
          )}
        </div>

        {expanded && (
          <div className="tree-children" style={{ marginLeft: indent + 18 }}>
            {classes.map((c, i) => (
              <div key={`c-${i}`} className="tree-symbol-class">
                ◆ {c.name}
              </div>
            ))}
            {functions.map((f, i) => (
              <div key={`f-${i}`} className="tree-symbol-fn">
                ƒ {f.name}
              </div>
            ))}
            {classes.length === 0 && functions.length === 0 && (
              <div className="tree-symbol-empty">(no parsed symbols)</div>
            )}
          </div>
        )}
      </div>
    );
  }

  return (
    <div>
      <div
        className="tree-node-row tree-node-dir"
        style={{ paddingLeft: indent + 10 }}
        onClick={() => setExpanded(e => !e)}
      >
        {expanded ? <FolderOpen size={14} className="tree-icon-folder" /> : <Folder size={14} className="tree-icon-folder" />}
        <span>{name}</span>
        <span className="tree-chevron">
          {expanded ? <ChevronDown size={12} /> : <ChevronRight size={12} />}
        </span>
      </div>
      {expanded && (
        <div className="tree-children" style={{ marginLeft: indent + 18 }}>
          {Object.keys(data).sort().map(childName => (
            <TreeNode
              key={childName}
              name={childName}
              data={data[childName]}
              isFile={!!data[childName].__isFile}
              depth={depth + 1}
            />
          ))}
        </div>
      )}
    </div>
  );
};

const StructureExplorer = () => {
  const [fileTree, setFileTree] = useState(null);
  const [loading, setLoading]   = useState(true);

  useEffect(() => {
    const fetch_ = async () => {
      try {
        const res = await axios.get(`${API_BASE_URL}/api/structure`);
        const rawFiles = res.data;
        const root = {};
        Object.keys(rawFiles).forEach(path => {
          const parts = path.split('/');
          let cur = root;
          parts.forEach((part, idx) => {
            if (idx === parts.length - 1) {
              cur[part] = { ...rawFiles[path], __isFile: true };
            } else {
              if (!cur[part]) cur[part] = {};
              cur = cur[part];
            }
          });
        });
        setFileTree(root);
      } catch (err) {
        console.error('Failed to fetch structure', err);
      } finally {
        setLoading(false);
      }
    };
    fetch_();
  }, []);

  return (
    <div className="structure-container">
      <div className="section-header">
        <FolderTree size={17} style={{ color: 'var(--accent)' }} />
        <h2>Repository Explorer</h2>
        <span className="section-badge">AST View</span>
      </div>

      {loading && (
        <div className="empty-state-text">Loading file tree…</div>
      )}
      {!loading && !fileTree && (
        <div className="empty-state-text">No structure available.</div>
      )}
      {!loading && fileTree && (
        <div className="tree-scroll">
          {Object.keys(fileTree).sort().map(name => (
            <TreeNode
              key={name}
              name={name}
              data={fileTree[name]}
              isFile={!!fileTree[name].__isFile}
              depth={0}
            />
          ))}
        </div>
      )}
    </div>
  );
};

export default StructureExplorer;
