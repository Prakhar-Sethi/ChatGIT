import React, { useState, useEffect, useRef } from 'react';
import ForceGraph2D from 'react-force-graph-2d';
import axios from 'axios';
import { API_BASE_URL } from '../config';
import { Network } from 'lucide-react';

const CallGraph = () => {
  const [graphElements, setGraphElements] = useState({ nodes: [], links: [] });
  const [focusedNode, setFocusedNode]     = useState('Show All');
  const [nodeList, setNodeList]           = useState([]);
  const wrapperRef = useRef(null);
  const [size, setSize] = useState({ width: 800, height: 520 });

  useEffect(() => {
    axios.get(`${API_BASE_URL}/api/call_graph`)
      .then(res => {
        if (res.data.functions) setNodeList(['Show All', ...res.data.functions]);
      })
      .catch(err => console.error(err));
  }, []);

  useEffect(() => {
    const loadGraph = async () => {
      try {
        const res = await axios.post(`${API_BASE_URL}/api/call_graph/visualize`, { target: focusedNode });
        if (res.data.nodes) {
          const connections = res.data.edges.map(e => ({ source: e.source, target: e.target }));
          setGraphElements({ nodes: res.data.nodes, links: connections });
        }
      } catch (e) {
        console.error(e);
      }
    };
    loadGraph();
  }, [focusedNode]);

  useEffect(() => {
    const measure = () => {
      if (wrapperRef.current) {
        setSize({ width: wrapperRef.current.offsetWidth, height: 520 });
      }
    };
    measure();
    window.addEventListener('resize', measure);
    return () => window.removeEventListener('resize', measure);
  }, []);

  return (
    <div className="call-graph-container" ref={wrapperRef}>
      <div className="call-graph-header">
        <div className="call-graph-title">
          <Network size={16} style={{ color: 'var(--accent)' }} />
          <span>Call Graph Visualization</span>
        </div>
        <div className="call-graph-controls">
          <select
            className="call-graph-select"
            value={focusedNode}
            onChange={e => setFocusedNode(e.target.value)}
          >
            {nodeList.map(f => <option key={f} value={f}>{f}</option>)}
          </select>
          <div className="call-graph-stats">
            {graphElements.nodes.length} nodes &middot; {graphElements.links.length} edges
          </div>
        </div>
      </div>

      <div className="call-graph-canvas">
        <ForceGraph2D
          graphData={graphElements}
          nodeLabel="label"
          backgroundColor="#f9fafb"
          width={size.width}
          height={size.height}
          linkDirectionalArrowLength={3.5}
          linkDirectionalArrowRelPos={1}
          nodeCanvasObject={(node, ctx, globalScale) => {
            const label = node.label;
            const fontSize = 12 / globalScale;
            ctx.font = `${fontSize}px Inter, sans-serif`;
            const textWidth = ctx.measureText(label).width;
            const bckgDimensions = [textWidth, fontSize].map(n => n + fontSize * 0.2);

            ctx.beginPath();
            ctx.arc(node.x, node.y, 4, 0, 2 * Math.PI, false);
            ctx.fillStyle = node.id === focusedNode ? '#4f46e5' : '#6366f1';
            ctx.fill();

            ctx.textAlign = 'center';
            ctx.textBaseline = 'middle';
            ctx.fillStyle = '#111827';
            ctx.fillText(label, node.x, node.y + 7);

            node.__bckgDimensions = bckgDimensions;
          }}
          nodePointerAreaPaint={(node, color, ctx) => {
            ctx.fillStyle = color;
            const bckgDimensions = node.__bckgDimensions;
            bckgDimensions && ctx.fillRect(
              node.x - bckgDimensions[0] / 2,
              node.y - bckgDimensions[1] / 2,
              ...bckgDimensions
            );
          }}
          linkColor={() => 'rgba(99, 102, 241, 0.3)'}
        />
      </div>
    </div>
  );
};

export default CallGraph;
