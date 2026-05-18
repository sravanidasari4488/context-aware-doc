import { StrictMode } from 'react';
import { createRoot } from 'react-dom/client';
import ContextAwareDocQABot from './ContextAwareDocQABot.jsx';

createRoot(document.getElementById('root')).render(
  <StrictMode>
    <ContextAwareDocQABot />
  </StrictMode>,
);
