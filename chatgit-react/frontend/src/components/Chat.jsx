import React, { useState, useEffect, useRef, useCallback, memo } from 'react';
import axios from 'axios';
import { API_BASE_URL } from '../config';
import ReactMarkdown from 'react-markdown';
import { Prism as SyntaxHighlighter } from 'react-syntax-highlighter';
import { oneDark } from 'react-syntax-highlighter/dist/esm/styles/prism';
import { MessageSquare, Send } from 'lucide-react';

const TypingIndicator = () => (
  <div className="message assistant" style={{ animationDelay: '0s' }}>
    <div className="msg-label">
      <span className="msg-label-dot" />
      ChatGIT
    </div>
    <div className="typing-indicator">
      <div className="typing-dot" />
      <div className="typing-dot" />
      <div className="typing-dot" />
    </div>
  </div>
);

const lineNumberRegex = /^\s*(\d+)\s*\|\s*/;

const CodeBlock = memo(({ className, children }) => {
  const langMatch = /language-(\w+)/.exec(className || '');
  const raw = String(children).replace(/\n$/, '');

  let startLine = 1;
  let code = raw;
  if (lineNumberRegex.test(raw)) {
    const m = raw.match(lineNumberRegex);
    if (m) startLine = parseInt(m[1], 10);
    code = raw.split('\n').map(l => l.replace(lineNumberRegex, '')).join('\n');
  }

  if (!langMatch) {
    return (
      <code className="inline-code">{children}</code>
    );
  }

  return (
    <SyntaxHighlighter
      language={langMatch[1]}
      style={oneDark}
      showLineNumbers
      startingLineNumber={startLine}
      wrapLines
      customStyle={{ margin: 0, borderRadius: 6, fontSize: '12.5px', lineHeight: '1.55' }}
    >
      {code}
    </SyntaxHighlighter>
  );
});

const MessageBubble = memo(({ role, content }) => (
  <div className={`message ${role}`}>
    <div className="msg-label">
      <span className="msg-label-dot" />
      {role === 'user' ? 'You' : 'ChatGIT'}
    </div>
    <div className="msg-bubble">
      <ReactMarkdown
        components={{
          code({ node, inline, className, children, ...props }) {
            return inline
              ? <code className="inline-code">{children}</code>
              : <CodeBlock className={className}>{children}</CodeBlock>;
          },
          strong({ children }) {
            return <strong className="md-strong">{children}</strong>;
          },
        }}
      >
        {content}
      </ReactMarkdown>
    </div>
  </div>
));

const ChatInput = memo(({ onSend, waiting, message, setMessage, onKey }) => (
  <div className="chat-input-area">
    <div className="chat-input-wrap">
      <textarea
        className="chat-input"
        rows={1}
        value={message}
        onChange={e => setMessage(e.target.value)}
        onKeyDown={onKey}
        placeholder="Ask about the code… (Enter to send, Shift+Enter for newline)"
        autoComplete="off"
        spellCheck={false}
      />
    </div>
    <button className="chat-send-btn" onClick={onSend} disabled={waiting || !message.trim()}>
      <Send size={16} />
    </button>
  </div>
));

const Chat = ({ chatLog, codeEnhancement }) => {
  const [conversation, setConversation] = useState([]);
  const [message, setMessage] = useState('');
  const [waiting, setWaiting] = useState(false);
  const bottomRef = useRef(null);

  useEffect(() => { if (chatLog) setConversation(chatLog); }, [chatLog]);
  useEffect(() => { bottomRef.current?.scrollIntoView({ behavior: 'smooth' }); }, [conversation, waiting]);

  const send = useCallback(async () => {
    if (!message.trim() || waiting) return;
    const userMsg = { role: 'user', content: message };
    setConversation(prev => [...prev, userMsg]);
    setMessage('');
    setWaiting(true);
    try {
      const r = await axios.post(`${API_BASE_URL}/api/chat`, {
        message: userMsg.content,
        enhance_code: codeEnhancement,
      });
      setConversation(r.data.history);
    } catch (err) {
      const detail = err.response?.data?.detail || err.message || 'Could not reach the AI service. Please check the server.';
      setConversation(prev => [...prev, {
        role: 'assistant',
        content: `**Error:** ${detail}`,
      }]);
    } finally {
      setWaiting(false);
    }
  }, [message, waiting, codeEnhancement]);

  const onKey = useCallback((e) => {
    if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); send(); }
  }, [send]);

  return (
    <div>
      <div className="chat-section-header">
        <MessageSquare size={18} style={{ color: 'var(--accent)' }} />
        <h2>Chat</h2>
        <span className="section-badge">Multi-turn · Session memory · Intent routing</span>
      </div>

      <div className="chat-container">
        <div className="chat-messages">
          {conversation.length === 0 && !waiting && (
            <div className="chat-empty">
              <div className="chat-empty-icon">
                <MessageSquare size={32} />
              </div>
              <div>Ask anything about the repository</div>
              <div className="chat-empty-hint">
                Try: "Explain the main entry point" · "Where is auth handled?" · "Debug the error handler"
              </div>
            </div>
          )}
          {conversation.map((m, i) => (
            <MessageBubble key={i} role={m.role} content={m.content} />
          ))}
          {waiting && <TypingIndicator />}
          <div ref={bottomRef} />
        </div>

        <ChatInput onSend={send} waiting={waiting} message={message} setMessage={setMessage} onKey={onKey} />
      </div>
    </div>
  );
};

export default Chat;
