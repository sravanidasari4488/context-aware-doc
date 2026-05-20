import {
  useState,
  useEffect,
  useRef,
  useCallback,
  useMemo,
} from 'react';
/** @typedef {{ id: string; text: string; page: number; startChar: number; endChar: number; docName: string; chunkIndex: number; tfidf?: Record<string, number> }} Chunk */
/** @typedef {{ id: string; name: string; chunks: Chunk[] }} Document */

const STOPWORDS = new Set([
  'the',
  'a',
  'an',
  'is',
  'in',
  'of',
  'to',
  'and',
  'or',
  'for',
  'with',
  'on',
  'at',
  'by',
  'from',
  'as',
  'be',
  'this',
  'that',
  'it',
  'are',
  'was',
  'were',
  'has',
  'have',
  'had',
  'not',
  'but',
  'so',
  'if',
  'do',
  'did',
  'does',
  'can',
  'will',
  'would',
  'could',
  'should',
  'its',
  'their',
  'they',
  'we',
  'you',
  'he',
  'she',
  'him',
  'her',
  'our',
  'my',
  'your',
  'which',
  'who',
  'what',
  'when',
  'where',
  'how',
  'all',
  'one',
  'more',
  'also',
  'than',
  'then',
  'into',
  'about',
  'up',
  'out',
  'been',
  'there',
  'no',
  'may',
  'just',
  'over',
  'such',
  'after',
  'before',
  'other',
  'new',
  'only',
  'these',
  'those',
  'through',
  'during',
  'between',
  'while',
  'each',
  'both',
  'few',
  'some',
  'any',
  'most',
  'same',
  'different',
  'used',
  'using',
  'made',
  'make',
  'many',
  'much',
  'first',
  'last',
  'long',
  'little',
  'own',
  'right',
  'still',
  'even',
  'back',
  'way',
  'well',
  'now',
  'known',
  'called',
  'include',
  'including',
  'included',
  'within',
  'without',
  'along',
  'across',
  'around',
  'whether',
  'either',
  'neither',
  'every',
  'never',
  'always',
  'often',
  'usually',
  'generally',
  'however',
  'therefore',
  'thus',
  'hence',
  'since',
  'because',
  'although',
  'though',
  'whereas',
]);

const CHUNK_SIZE = 600;
const OVERLAP = 120;
const PDF_JS_URL =
  'https://cdnjs.cloudflare.com/ajax/libs/pdf.js/3.11.174/pdf.min.js';
const PDF_WORKER_URL =
  'https://cdnjs.cloudflare.com/ajax/libs/pdf.js/3.11.174/pdf.worker.min.js';

const MIN_RETRIEVAL_SIMILARITY = 0.02;

const FOLLOW_UP_PHRASES = [
  'elaborate',
  'tell me more',
  'explain further',
  'what did you mean',
  'can you explain',
  'more detail',
  'go on',
  'continue',
  'and then',
  'what about',
  'how about',
  'why is that',
  'how so',
];

const GEMINI_MODEL_FALLBACKS = [
  'gemini-2.5-flash-lite',
  'gemini-2.5-flash',
];
const GEMINI_MIN_INTERVAL_MS = 5000;
const GEMINI_RATE_LIMIT_RETRY_MS = 5000;
const GEMINI_MAX_RETRIES = 4;

let lastGeminiCallAt = 0;
let geminiRequestQueue = Promise.resolve();

function sleep(ms) {
  return new Promise((resolve) => {
    setTimeout(resolve, ms);
  });
}

function enqueueGeminiRequest(task) {
  const run = geminiRequestQueue.then(() => task());
  geminiRequestQueue = run.catch(() => {});
  return run;
}

function getGeminiCooldownRemainingMs() {
  if (lastGeminiCallAt <= 0) return 0;
  return Math.max(0, lastGeminiCallAt + GEMINI_MIN_INTERVAL_MS - Date.now());
}

function geminiModelsToTry() {
  const preferred = import.meta.env.VITE_GEMINI_MODEL;
  const list = preferred
    ? [preferred, ...GEMINI_MODEL_FALLBACKS]
    : [...GEMINI_MODEL_FALLBACKS];
  return [...new Set(list.filter(Boolean))];
}

function isGeminiRateLimitError(status, message) {
  if (status === 429) return true;
  const m = String(message).toLowerCase();
  return (
    m.includes('quota') ||
    m.includes('rate limit') ||
    m.includes('resource_exhausted')
  );
}

/** Daily free-tier caps do not clear after a short wait — try another model instead. */
function isGeminiDailyQuotaError(message, data) {
  const m = String(message).toLowerCase();
  if (m.includes('per day') || m.includes('perday')) return true;
  const details = data?.error?.details;
  if (!Array.isArray(details)) return false;
  for (const d of details) {
    const violations = d.violations;
    if (!Array.isArray(violations)) continue;
    for (const v of violations) {
      const id = String(v.quotaId || '');
      if (id.includes('PerDay') || id.includes('PerDayPer')) return true;
    }
  }
  return false;
}

function isGeminiModelNotFoundError(status, message) {
  const m = String(message).toLowerCase();
  return (
    status === 404 ||
    m.includes('not found') ||
    m.includes('not supported for generatecontent')
  );
}

function uuid() {
  if (typeof crypto !== 'undefined' && crypto.randomUUID) {
    return crypto.randomUUID();
  }
  return `chunk-${Date.now()}-${Math.random().toString(36).slice(2, 11)}`;
}

/**
 * Exact chunking rules: chunkSize 600, overlap 120, prefer sentence ends (. ! ?).
 * @param {string} text
 * @param {string} docName
 * @param {number} page
 * @returns {Omit<Chunk, 'endChar'>[]}
 */
function chunkText(text, docName, page) {
  const chunks = [];
  if (!text || text.length === 0) {
    return chunks;
  }

  const sentenceEndRe = /[.!?](?:\s|$)/g;
  let start = 0;
  let chunkIndex = 0;

  while (start < text.length) {
    let end = Math.min(start + CHUNK_SIZE, text.length);

    if (end < text.length) {
      const searchFrom = Math.max(start + Math.floor(CHUNK_SIZE * 0.35), start);
      let best = -1;
      sentenceEndRe.lastIndex = 0;
      let m;
      while ((m = sentenceEndRe.exec(text)) !== null) {
        const boundaryEnd = m.index + 1;
        if (boundaryEnd <= searchFrom) continue;
        if (boundaryEnd > end) break;
        best = boundaryEnd;
      }
      if (best !== -1) {
        end = best;
      }
    }

    const slice = text.slice(start, end).trim();
    if (slice.length === 0) {
      if (end >= text.length) break;
      start = Math.max(start + 1, end);
      continue;
    }

    const startChar = start;
    chunks.push({
      id: uuid(),
      text: slice,
      docName,
      page,
      chunkIndex,
      startChar,
    });
    chunkIndex += 1;

    if (end >= text.length) break;

    let nextStart = end - OVERLAP;
    if (nextStart <= start) {
      nextStart = start + 1;
    }
    start = nextStart;
  }

  return chunks;
}

function loadScript(src) {
  return new Promise((resolve, reject) => {
    const existing = document.querySelector(`script[src="${src}"]`);
    if (existing) {
      if (existing.getAttribute('data-loaded') === 'true') {
        resolve();
        return;
      }
      existing.addEventListener('load', () => resolve(), { once: true });
      existing.addEventListener('error', () => reject(new Error(src)), {
        once: true,
      });
      return;
    }
    const s = document.createElement('script');
    s.src = src;
    s.async = true;
    s.onload = () => {
      s.setAttribute('data-loaded', 'true');
      resolve();
    };
    s.onerror = () => reject(new Error(`Failed to load script: ${src}`));
    document.head.appendChild(s);
  });
}

async function ensurePdfJs() {
  await loadScript(PDF_JS_URL);
  const pdfjsLib = window.pdfjsLib;
  if (!pdfjsLib) {
    throw new Error('pdf.js did not expose pdfjsLib on window');
  }
  pdfjsLib.GlobalWorkerOptions.workerSrc = PDF_WORKER_URL;
  return pdfjsLib;
}

function readFileAsText(file) {
  return new Promise((resolve, reject) => {
    const fr = new FileReader();
    fr.onload = () => resolve(String(fr.result ?? ''));
    fr.onerror = () => reject(fr.error);
    fr.readAsText(file);
  });
}

async function parseTxtFile(file) {
  const fullText = await readFileAsText(file);
  return [{ page: 1, text: fullText }];
}

/**
 * Preserve line breaks from PDF layout so definitions stay with their headings.
 * @param {Array<{ str?: string; transform?: number[]; hasEOL?: boolean }>} items
 */
function pdfPageItemsToText(items) {
  let text = '';
  let lastY = null;
  for (const item of items) {
    if (!item || !('str' in item) || !item.str) continue;
    const y = item.transform?.[5];
    if (
      lastY != null &&
      y != null &&
      Math.abs(y - lastY) > 4
    ) {
      text += '\n';
    } else if (text.length > 0 && !text.endsWith('\n') && !text.endsWith(' ')) {
      text += ' ';
    }
    text += item.str;
    if (item.hasEOL) {
      text += '\n';
    }
    lastY = y ?? lastY;
  }
  return text.replace(/\n{3,}/g, '\n\n').trim();
}

async function parsePdfFile(file) {
  const pdfjsLib = await ensurePdfJs();
  const data = await file.arrayBuffer();
  const pdf = await pdfjsLib.getDocument({ data }).promise;
  const pages = [];
  for (let p = 1; p <= pdf.numPages; p += 1) {
    const page = await pdf.getPage(p);
    const tc = await page.getTextContent();
    const text = pdfPageItemsToText(tc.items);
    pages.push({ page: p, text });
  }
  return pages;
}

/**
 * @param {File[]} files
 * @param {(s: string) => void} onStatus
 * @returns {Promise<{ builtDocs: Document[]; totalPages: number }>}
 */
async function ingestFilesToBuiltDocs(files, onStatus) {
  const builtDocs = [];
  let totalPages = 0;
  const n = files.length;
  for (let i = 0; i < n; i += 1) {
    const file = files[i];
    const isPdf =
      file.type === 'application/pdf' || /\.pdf$/i.test(file.name);
    onStatus(
      `Step 1/3: Parsing ${isPdf ? 'PDF' : 'text'} (${i + 1}/${n})…`,
    );
    const pageSegments = isPdf
      ? await parsePdfFile(file)
      : await parseTxtFile(file);
    totalPages += pageSegments.length;
    onStatus(`Step 2/3: Chunking "${file.name}" (${i + 1}/${n})…`);
    const allRawChunks = [];
    for (const seg of pageSegments) {
      allRawChunks.push(...chunkText(seg.text, file.name, seg.page));
    }
    const chunks = allRawChunks.map((c) => ({
      ...c,
      endChar: c.startChar + c.text.length,
    }));
    builtDocs.push({
      id: uuid(),
      name: file.name,
      chunks,
    });
  }
  onStatus('Step 3/3: Building search index…');
  return { builtDocs, totalPages };
}

function formatBytes(n) {
  if (n < 1024) return `${n} B`;
  if (n < 1024 * 1024) return `${(n / 1024).toFixed(1)} KB`;
  return `${(n / (1024 * 1024)).toFixed(1)} MB`;
}

/**
 * @param {string} text
 * @returns {string[]}
 */
function tokenize(text) {
  return String(text)
    .toLowerCase()
    .replace(/[^a-z0-9]+/g, ' ')
    .split(/\s+/)
    .filter((w) => w.length > 0 && !STOPWORDS.has(w));
}

/**
 * @param {string[]} tokens
 * @returns {Record<string, number>}
 */
function termFrequencyMap(tokens) {
  const freq = {};
  for (const t of tokens) {
    freq[t] = (freq[t] || 0) + 1;
  }
  return freq;
}

/**
 * Document frequency: number of chunks containing each term.
 * @param {Chunk[]} chunks
 * @returns {Record<string, number>}
 */
function documentFrequency(chunks) {
  const df = {};
  for (const chunk of chunks) {
    const seen = new Set(tokenize(chunk.text));
    for (const t of seen) {
      df[t] = (df[t] || 0) + 1;
    }
  }
  return df;
}

/**
 * @param {Record<string, number>} df
 * @param {number} N chunk count
 * @returns {Record<string, number>}
 */
function inverseDocumentFrequency(df, N) {
  const idf = {};
  for (const [term, d] of Object.entries(df)) {
    idf[term] = Math.log(1 + N / Math.max(d, 1));
  }
  return idf;
}

/**
 * Sparse TF-IDF vector: { word: score }
 * @param {string} text
 * @param {Record<string, number>} idf
 */
function textToTfidfVector(text, idf) {
  const tokens = tokenize(text);
  if (tokens.length === 0) return {};
  const tf = termFrequencyMap(tokens);
  const total = tokens.length;
  /** @type {Record<string, number>} */
  const vec = {};
  for (const [term, cnt] of Object.entries(tf)) {
    if (idf[term] == null) continue;
    vec[term] = (cnt / total) * idf[term];
  }
  return vec;
}

/**
 * Attach sparse TF-IDF to each chunk; returns corpus IDF map for queries.
 * @param {Chunk[]} allChunks
 * @returns {Record<string, number>}
 */
function buildTfidfIndexForChunks(allChunks) {
  const N = allChunks.length;
  if (N === 0) return {};
  const df = documentFrequency(allChunks);
  const idf = inverseDocumentFrequency(df, N);
  for (const chunk of allChunks) {
    chunk.tfidf = textToTfidfVector(chunk.text, idf);
  }
  return idf;
}

/**
 * @param {Record<string, number>} vecA
 * @param {Record<string, number>} vecB
 * @returns {number} cosine similarity clamped to [0, 1]
 */
function cosineSimilarity(vecA, vecB) {
  let dot = 0;
  for (const w of Object.keys(vecA)) {
    if (vecB[w] != null) dot += vecA[w] * vecB[w];
  }
  let magA = 0;
  for (const w of Object.keys(vecA)) {
    magA += vecA[w] * vecA[w];
  }
  let magB = 0;
  for (const w of Object.keys(vecB)) {
    magB += vecB[w] * vecB[w];
  }
  magA = Math.sqrt(magA);
  magB = Math.sqrt(magB);
  if (magA === 0 || magB === 0) return 0;
  const raw = dot / (magA * magB);
  return Math.min(1, Math.max(0, raw));
}

/**
 * @param {number[]} topScores sorted descending, similarity values
 */
function confidenceFromTopScores(topScores) {
  const k = Math.min(3, topScores.length);
  if (k === 0) {
    return {
      avg: 0,
      label: 'Low confidence — answer may not be in document',
    };
  }
  const avg =
    topScores.slice(0, k).reduce((sum, s) => sum + s, 0) / k;
  if (avg > 0.3) {
    return { avg, label: 'High confidence' };
  }
  if (avg >= 0.1) {
    return { avg, label: 'Medium confidence' };
  }
  return {
    avg,
    label: 'Low confidence — answer may not be in document',
  };
}

function isFollowUpQuestion(query) {
  const lower = String(query).toLowerCase();
  return FOLLOW_UP_PHRASES.some((phrase) => lower.includes(phrase));
}

/**
 * @param {{ role: string; content: string }[]} messages
 * @returns {{ role: string; content: string } | null}
 */
function getLastAssistantMessage(messages) {
  for (let i = messages.length - 1; i >= 0; i -= 1) {
    const m = messages[i];
    if (m.role === 'assistant' && m.content?.trim()) {
      return m;
    }
  }
  return null;
}

/** Key topic tokens from the start of the prior assistant answer (for retrieval). */
function extractRetrievalTopicsFromAssistantMessage(content) {
  const excerpt = String(content)
    .replace(/\*\*/g, '')
    .slice(0, 200)
    .trim();
  const tokens = tokenize(excerpt);
  if (tokens.length > 0) return tokens.join(' ');
  return excerpt;
}

/**
 * @param {string} userQuery
 * @param {{ role: string; content: string }[]} chatHistory messages before the current user turn
 * @returns {{ retrievalQuery: string; isFollowUp: boolean }}
 */
function resolveRetrievalQuery(userQuery, chatHistory) {
  if (!isFollowUpQuestion(userQuery)) {
    return { retrievalQuery: userQuery, isFollowUp: false };
  }
  const lastAssistant = getLastAssistantMessage(chatHistory);
  if (!lastAssistant) {
    return { retrievalQuery: userQuery, isFollowUp: false };
  }
  const topics = extractRetrievalTopicsFromAssistantMessage(
    lastAssistant.content,
  );
  if (!topics.trim()) {
    return { retrievalQuery: userQuery, isFollowUp: false };
  }
  return { retrievalQuery: topics, isFollowUp: true };
}

/** Normalize query text before retrieval; typos are kept as typed. */
function preprocessQueryForRetrieval(query) {
  return String(query)
    .toLowerCase()
    .trim()
    .replace(/\s+/g, ' ');
}

/**
 * For query tokens missing from the corpus vocabulary, add vocab terms that
 * share the same 4-character prefix (e.g. "nanotub" → "nanotube", "nanotubes").
 * @param {string} query
 * @param {Record<string, number>} vocabularyIdf
 * @returns {string}
 */
function expandQueryWithFuzzyVocabularyTokens(query, vocabularyIdf) {
  const tokens = tokenize(query);
  if (tokens.length === 0) return query;

  /** @type {Set<string>} */
  const added = new Set();
  for (const word of tokens) {
    if (vocabularyIdf[word] != null) continue;
    if (word.length < 4) continue;
    const prefix = word.slice(0, 4);
    for (const term of Object.keys(vocabularyIdf)) {
      if (term !== word && term.startsWith(prefix)) {
        added.add(term);
      }
    }
  }
  if (added.size === 0) return query;
  return `${query} ${[...added].join(' ')}`;
}

function expandQueryForRetrieval(query) {
  const base = String(query);
  const lower = base.toLowerCase();
  const extras = [];
  if (/\btypes?\b/.test(lower)) {
    extras.push('kinds categories list describe definition');
  }
  if (/nanotube|\bcnts?\b/i.test(lower)) {
    extras.push(
      'SWCNT MWCNT single-walled multi-walled single walled multi walled graphene cylindrical',
    );
  }
  return extras.length ? `${base} ${extras.join(' ')}` : base;
}

/**
 * Include adjacent chunks on the same page so body text near a heading is not missed.
 * @param {{ chunk: Chunk; similarity: number }[]} results
 * @param {Chunk[]} allChunks
 */
function expandRetrievedWithNeighbors(results, allChunks, maxExtra = 8) {
  const seen = new Set(results.map((r) => r.chunk.id));
  const extras = [];
  for (const r of results) {
    const { docName, page, chunkIndex } = r.chunk;
    for (const delta of [-1, 1]) {
      if (extras.length >= maxExtra) break;
      const sibling = allChunks.find(
        (c) =>
          c.docName === docName &&
          c.page === page &&
          c.chunkIndex === chunkIndex + delta,
      );
      if (sibling && !seen.has(sibling.id)) {
        seen.add(sibling.id);
        extras.push({
          chunk: sibling,
          similarity: r.similarity * 0.92,
        });
      }
    }
  }
  const merged = [...results, ...extras];
  merged.sort((a, b) => b.similarity - a.similarity);
  return merged;
}

/**
 * @param {string} query
 * @param {Chunk[]} allChunks chunks with `.tfidf` populated
 * @param {number} [topK=7]
 * @param {Record<string, number>} [idfMap] corpus IDF from indexing (required for query vectors)
 */
function retrieveRelevantChunks(query, allChunks, topK = 7, idfMap) {
  if (!idfMap || allChunks.length === 0) {
    return { results: [], confidence: confidenceFromTopScores([]) };
  }
  const normalizedQuery = preprocessQueryForRetrieval(query);
  const fuzzyQuery = expandQueryWithFuzzyVocabularyTokens(
    normalizedQuery,
    idfMap,
  );
  const searchQuery = expandQueryForRetrieval(fuzzyQuery);
  const qVec = textToTfidfVector(searchQuery, idfMap);
  const scored = allChunks.map((chunk) => {
    const sim = cosineSimilarity(qVec, chunk.tfidf || {});
    return { chunk, similarity: sim };
  });
  const filtered = scored.filter(
    (s) => s.similarity >= MIN_RETRIEVAL_SIMILARITY,
  );
  filtered.sort((a, b) => b.similarity - a.similarity);
  const confidence = confidenceFromTopScores(
    filtered.slice(0, 3).map((r) => r.similarity),
  );
  const core = filtered.slice(0, topK);
  const results = expandRetrievedWithNeighbors(core, allChunks).slice(0, 12);
  return { results, confidence };
}

/**
 * @param {string} question
 * @param {{ text: string; docName: string; page: number }[]} relevantChunks
 * @param {{ role: string; content: string }[]} chatHistory
 * @param {{ onWait?: (label: string) => void; onWaitEnd?: () => void; onApiStart?: () => void }} [callbacks]
 * @param {{ isFollowUp?: boolean }} [options]
 * @returns {Promise<string>}
 */
async function answerQuestion(
  question,
  relevantChunks,
  chatHistory,
  callbacks = {},
  options = {},
) {
  const { onWait, onWaitEnd, onApiStart } = callbacks;
  const { isFollowUp = false } = options;
  const apiKey = import.meta.env.VITE_GEMINI_API_KEY;

  return enqueueGeminiRequest(async () => {
    const contextText = relevantChunks
      .map(
        (c, i) =>
          `[Source ${i + 1} | ${c.docName} | Page ${c.page}]\n${c.text}`,
      )
      .join('\n\n---\n\n');

    const followUpInstruction = isFollowUp
      ? `The user is asking a follow-up question about your previous response. Use the conversation history to understand context and elaborate on what was previously discussed.

`
      : '';

    const systemPrompt = `You are a document analysis assistant. Answer questions using ONLY the document excerpts below.

${followUpInstruction}RULES:
- Read every excerpt carefully. If any excerpt contains facts, definitions, measurements, or lists that answer the question (even partially), you MUST include them in your answer.
- When the question asks for "types", list each type named in the excerpts with its description from the text.
- Cite sources (e.g., "According to Source 2...").
- Say "I couldn't find information about this in the uploaded document(s)." ONLY if no excerpt contains any substantive content related to the question.
- Do not claim the excerpts only "mention" a topic without details if the text actually includes definitions or specifications.
- Never use outside knowledge.

DOCUMENT EXCERPTS:
${contextText}`;

    const historyMsgs = chatHistory
      .filter((m) => m.role === 'user' || m.role === 'assistant')
      .map((m) => ({ role: m.role, content: m.content }));

    const promptText =
      `${systemPrompt}\n\nConversation history:\n` +
      historyMsgs
        .slice(-6)
        .map((m) => `${m.role}: ${m.content}`)
        .join('\n') +
      `\n\nUser question: ${question}`;

    const elapsed = Date.now() - lastGeminiCallAt;
    if (lastGeminiCallAt > 0 && elapsed < GEMINI_MIN_INTERVAL_MS) {
      onWait?.('Thinking...');
      await sleep(GEMINI_MIN_INTERVAL_MS - elapsed);
      onWaitEnd?.();
    }

    const models = geminiModelsToTry();
    let apiStarted = false;
    let sawDailyQuota = false;
    let sawRateLimit = false;

    for (const model of models) {
      for (let attempt = 0; attempt <= GEMINI_MAX_RETRIES; attempt += 1) {
        lastGeminiCallAt = Date.now();
        if (!apiStarted) {
          onApiStart?.();
          apiStarted = true;
        }

        const response = await fetch(
          `https://generativelanguage.googleapis.com/v1beta/models/${model}:generateContent?key=${apiKey}`,
          {
            method: 'POST',
            headers: {
              'Content-Type': 'application/json',
            },
            body: JSON.stringify({
              contents: [
                {
                  parts: [{ text: promptText }],
                },
              ],
              generationConfig: {
                maxOutputTokens: 1000,
                temperature: 0.2,
              },
            }),
          },
        );

        const data = await response.json();
        if (response.ok) {
          const answer = data.candidates?.[0]?.content?.parts?.[0]?.text;
          if (answer != null && answer !== '') {
            return answer;
          }
          const finishReason = data.candidates?.[0]?.finishReason;
          if (finishReason === 'SAFETY' || finishReason === 'RECITATION') {
            return "I couldn't produce an answer for that question based on the document content. Try rephrasing your question.";
          }
          break;
        }

        const errMsg =
          data.error?.message || `Request failed (${response.status})`;

        if (isGeminiModelNotFoundError(response.status, errMsg)) {
          break;
        }

        if (isGeminiRateLimitError(response.status, errMsg)) {
          if (isGeminiDailyQuotaError(errMsg, data)) {
            sawDailyQuota = true;
            break;
          }
          sawRateLimit = true;
          if (attempt < GEMINI_MAX_RETRIES) {
            onWait?.('Just a moment...');
            await sleep(GEMINI_RATE_LIMIT_RETRY_MS);
            onWaitEnd?.();
            continue;
          }
          break;
        }

        break;
      }
    }

    if (sawDailyQuota && !sawRateLimit) {
      return "Today's free limit for the main Gemini model on this API key is used up. I tried alternate models — if this persists, wait until tomorrow (quota resets daily) or create a new key in Google AI Studio.";
    }
    if (sawDailyQuota || sawRateLimit) {
      return "I'm having a little trouble right now — please try your question again in a minute.";
    }

    return "I couldn't generate an answer just now. Please try again shortly.";
  });
}

function formatMessageTime(ts) {
  try {
    return new Date(ts).toLocaleTimeString([], {
      hour: '2-digit',
      minute: '2-digit',
      hour12: false,
    });
  } catch {
    return '';
  }
}

function MessageRichText({ text, className }) {
  const parts = String(text).split(/(\*\*[^*]+\*\*)/g);
  return (
    <span className={className}>
      {parts.map((seg, i) => {
        if (seg.startsWith('**') && seg.endsWith('**')) {
          return (
            <strong key={i} className="font-semibold">
              {seg.slice(2, -2)}
            </strong>
          );
        }
        return <span key={i}>{seg}</span>;
      })}
    </span>
  );
}

function BotAvatar() {
  return (
    <div
      className="flex h-9 w-9 shrink-0 items-center justify-center rounded-full bg-indigo-600 text-white shadow-md shadow-indigo-900/40"
      aria-hidden
    >
      <svg className="h-5 w-5" fill="currentColor" viewBox="0 0 24 24">
        <path d="M12 2a2 2 0 012 2c0 .74-.4 1.39-1 1.73V7h1a7 7 0 017 7h1v1a2 2 0 01-2 2H5a2 2 0 01-2-2v-1a7 7 0 017-7h1V5.73c-.6-.34-1-.99-1-1.73a2 2 0 012-2zM7.5 13a1.5 1.5 0 100 3 1.5 1.5 0 000-3zm9 0a1.5 1.5 0 100 3 1.5 1.5 0 000-3z" />
      </svg>
    </div>
  );
}

function ConfidenceBadge({ tier }) {
  if (tier === 'out_of_scope') {
    return (
      <span className="rounded-full bg-gray-600 px-2.5 py-0.5 text-xs font-medium text-gray-100">
        Out of scope
      </span>
    );
  }
  if (tier === 'high') {
    return (
      <span className="rounded-full bg-green-600 px-2.5 py-0.5 text-xs font-medium text-white">
        High confidence
      </span>
    );
  }
  if (tier === 'medium') {
    return (
      <span className="rounded-full bg-yellow-600 px-2.5 py-0.5 text-xs font-medium text-white">
        Medium confidence
      </span>
    );
  }
  return (
    <span className="rounded-full bg-red-600 px-2.5 py-0.5 text-xs font-medium text-white">
      Low confidence
    </span>
  );
}

function TypingIndicator({ label }) {
  return (
    <div className="flex items-center gap-2 pl-1" aria-live="polite">
      <div className="flex items-center gap-1.5">
        <span className="docqa-dot h-2 w-2 rounded-full bg-indigo-300" />
        <span className="docqa-dot docqa-dot-delay-1 h-2 w-2 rounded-full bg-indigo-300" />
        <span className="docqa-dot docqa-dot-delay-2 h-2 w-2 rounded-full bg-indigo-300" />
      </div>
      {label ? (
        <span className="text-xs text-gray-400">{label}</span>
      ) : null}
    </div>
  );
}

function SourceChunkItem({ source }) {
  const [fullOpen, setFullOpen] = useState(false);
  const full = source.fullText ?? source.preview;
  return (
    <li className="border-l-4 border-yellow-400 bg-gray-900/60 py-2 pl-3 pr-2 text-xs text-gray-300">
      <p className="font-medium text-gray-200">
        {source.docName} · Page {source.page}
      </p>
      <p className="mt-1 leading-relaxed text-gray-400">
        {fullOpen ? full : source.preview}
      </p>
      {full &&
      String(full).length > String(source.preview ?? '').length ? (
        <button
          type="button"
          onClick={() => setFullOpen((v) => !v)}
          className="mt-1.5 text-[11px] font-medium text-indigo-400 hover:text-indigo-300"
        >
          {fullOpen ? 'Show preview' : 'Show full chunk'}
        </button>
      ) : null}
    </li>
  );
}

function CollapsibleSources({ sources }) {
  const [open, setOpen] = useState(false);
  if (!sources || sources.length === 0) return null;
  const list = sources.slice(0, 3);
  return (
    <div className="mt-2 border-t border-gray-700/80 pt-2">
      <button
        type="button"
        onClick={() => setOpen((o) => !o)}
        className="flex w-full items-center justify-between rounded-lg px-1 py-1.5 text-left text-xs font-medium text-indigo-300 hover:bg-gray-700/50"
        aria-expanded={open}
      >
        <span>Sources ({list.length})</span>
        <span className="text-gray-500">{open ? '▼' : '▶'}</span>
      </button>
      {open && (
        <ul className="mt-2 space-y-2">
          {list.map((s, i) => (
            <SourceChunkItem
              key={`${s.docName}-${s.page}-${i}`}
              source={s}
            />
          ))}
        </ul>
      )}
    </div>
  );
}

function DocqaKeyframes() {
  return (
    <style>
      {`
        @keyframes docqa-gradient-move {
          0% { transform: translate(0, 0) scale(1); opacity: 0.35; }
          50% { transform: translate(-4%, 3%) scale(1.03); opacity: 0.55; }
          100% { transform: translate(0, 0) scale(1); opacity: 0.35; }
        }
        @keyframes docqa-dot-pulse {
          0%, 100% { opacity: 0.35; transform: scale(0.9); }
          50% { opacity: 1; transform: scale(1.15); }
        }
        .docqa-gradient-bg {
          animation: docqa-gradient-move 14s ease-in-out infinite;
        }
        .docqa-dot {
          animation: docqa-dot-pulse 1.1s ease-in-out infinite;
        }
        .docqa-dot-delay-1 { animation-delay: 0.18s; }
        .docqa-dot-delay-2 { animation-delay: 0.36s; }
      `}
    </style>
  );
}

function HelpModal({ open, onClose }) {
  useEffect(() => {
    if (!open) return;
    const onKey = (e) => {
      if (e.key === 'Escape') onClose();
    };
    window.addEventListener('keydown', onKey);
    return () => window.removeEventListener('keydown', onKey);
  }, [open, onClose]);

  if (!open) return null;
  return (
    <div
      className="fixed inset-0 z-[100] flex items-center justify-center bg-black/70 p-4 backdrop-blur-sm"
      role="dialog"
      aria-modal="true"
      aria-labelledby="help-title"
    >
      <button
        type="button"
        className="absolute inset-0 cursor-default"
        aria-label="Close help"
        onClick={onClose}
      />
      <div className="relative max-h-[85vh] w-full max-w-lg overflow-y-auto rounded-2xl border border-gray-700 bg-gray-900 p-6 text-left text-sm text-gray-300 shadow-2xl">
        <div className="mb-4 flex items-center justify-between">
          <h2 id="help-title" className="text-lg font-semibold text-white">
            How it works
          </h2>
          <button
            type="button"
            onClick={onClose}
            className="rounded-lg px-2 py-1 text-gray-400 hover:bg-gray-800 hover:text-white"
          >
            ✕
          </button>
        </div>
        <div className="space-y-4">
          <section>
            <p className="font-semibold text-indigo-300">Chunking</p>
            <p className="mt-1 leading-relaxed">
              Documents are split into <strong>600-character</strong> overlapping
              chunks (<strong>120-char overlap</strong>) at sentence boundaries,
              preserving context across chunk edges.
            </p>
          </section>
          <section>
            <p className="font-semibold text-indigo-300">Retrieval</p>
            <p className="mt-1 leading-relaxed">
              Each chunk and query is converted to a <strong>TF-IDF</strong>{' '}
              vector. At query time, <strong>cosine similarity</strong> ranks
              chunks by relevance. <strong>Top 5</strong> chunks are selected.
            </p>
          </section>
          <section>
            <p className="font-semibold text-indigo-300">Generation</p>
            <p className="mt-1 leading-relaxed">
              Retrieved chunks are injected into the model context with strict
              instructions to answer only from provided sources. Conversation
              history (<strong>last 6 turns</strong>) is included for follow-up
              questions.
            </p>
          </section>
          <section>
            <p className="font-semibold text-indigo-300">Confidence</p>
            <p className="mt-1 leading-relaxed">
              Based on average cosine similarity of the{' '}
              <strong>top 3</strong> retrieved chunks.
            </p>
          </section>
        </div>
      </div>
    </div>
  );
}

function UploadIcon() {
  return (
    <svg
      className="mx-auto h-12 w-12 text-indigo-400"
      fill="none"
      stroke="currentColor"
      viewBox="0 0 24 24"
      aria-hidden="true"
    >
      <path
        strokeLinecap="round"
        strokeLinejoin="round"
        strokeWidth={1.5}
        d="M7 16a4 4 0 01-.88-7.903A5 5 0 1115.9 6L16 6a5 5 0 011 9.9M15 13l-3-3m0 0l-3 3m3-3v12"
      />
    </svg>
  );
}

export default function ContextAwareDocQABot() {
  /** @type {[Document[], function]} */
  const [documents, setDocuments] = useState([]);
  const [chatHistory, setChatHistory] = useState([]);
  const [isProcessing, setIsProcessing] = useState(false);
  const [processingStatus, setProcessingStatus] = useState('');
  const [uploadedFileCount, setUploadedFileCount] = useState(0);
  /** 'upload' | 'processing' | 'chat' */
  const [view, setView] = useState('upload');
  const [pendingFiles, setPendingFiles] = useState([]);
  const [lastSummary, setLastSummary] = useState({
    chunkCount: 0,
    pageCount: 0,
  });
  /** Corpus IDF map for query TF-IDF (same weights as chunk indexing). */
  const [corpusIdf, setCorpusIdf] = useState(null);
  const [queryInput, setQueryInput] = useState('');
  const [isAwaitingClaude, setIsAwaitingClaude] = useState(false);
  const [isThinking, setIsThinking] = useState(false);
  const [pendingLabel, setPendingLabel] = useState('Thinking...');
  const [cooldownActive, setCooldownActive] = useState(false);
  const [cooldownDurationMs, setCooldownDurationMs] = useState(0);
  const [cooldownFill, setCooldownFill] = useState(false);
  const [inputReadyFlash, setInputReadyFlash] = useState(false);
  const [helpOpen, setHelpOpen] = useState(false);
  const fileInputRef = useRef(null);
  const mergeFileInputRef = useRef(null);
  const messageScrollRef = useRef(null);
  const sendLockRef = useRef(false);
  useEffect(() => {
    setUploadedFileCount(pendingFiles.length);
  }, [pendingFiles.length]);

  const resetToUpload = useCallback(() => {
    setView('upload');
    setDocuments([]);
    setPendingFiles([]);
    setUploadedFileCount(0);
    setChatHistory([]);
    setProcessingStatus('');
    setIsProcessing(false);
    setLastSummary({ chunkCount: 0, pageCount: 0 });
    setCorpusIdf(null);
    setQueryInput('');
    setIsAwaitingClaude(false);
    setIsThinking(false);
    setPendingLabel('Thinking...');
    setCooldownActive(false);
    setCooldownFill(false);
    setInputReadyFlash(false);
    if (fileInputRef.current) fileInputRef.current.value = '';
    if (mergeFileInputRef.current) mergeFileInputRef.current.value = '';
    setHelpOpen(false);
  }, []);

  const addFiles = useCallback((fileList) => {
    const arr = Array.from(fileList || []).filter(
      (f) =>
        f.type === 'application/pdf' ||
        f.type === 'text/plain' ||
        /\.pdf$/i.test(f.name) ||
        /\.txt$/i.test(f.name),
    );
    if (arr.length === 0) return;
    setPendingFiles((prev) => {
      const next = [...prev];
      for (const f of arr) {
        if (!next.some((x) => x.name === f.name && x.size === f.size)) {
          next.push(f);
        }
      }
      return next;
    });
  }, []);

  const onInputChange = useCallback(
    (e) => {
      addFiles(e.target.files);
    },
    [addFiles],
  );

  const onDragEnter = useCallback((e) => {
    e.preventDefault();
    e.stopPropagation();
  }, []);

  const onDragLeave = useCallback((e) => {
    e.preventDefault();
    e.stopPropagation();
  }, []);

  const onDragOver = useCallback((e) => {
    e.preventDefault();
    e.stopPropagation();
  }, []);

  const onDrop = useCallback(
    (e) => {
      e.preventDefault();
      e.stopPropagation();
      addFiles(e.dataTransfer.files);
    },
    [addFiles],
  );

  const clearChat = useCallback(() => {
    setChatHistory([
      {
        id: uuid(),
        role: 'assistant',
        content:
          'Chat cleared. Your documents are still indexed — ask a new question anytime.',
        timestamp: Date.now(),
      },
    ]);
    setQueryInput('');
  }, []);

  const removeFileAt = useCallback((index) => {
    setPendingFiles((prev) => prev.filter((_, i) => i !== index));
  }, []);

  const processDocuments = useCallback(async () => {
    if (pendingFiles.length === 0) return;

    setView('processing');
    setIsProcessing(true);
    setProcessingStatus('Step 1/3: Starting…');

    try {
      const { builtDocs, totalPages } = await ingestFilesToBuiltDocs(
        pendingFiles,
        setProcessingStatus,
      );

      const flatChunks = builtDocs.flatMap((d) => d.chunks);
      const idf = buildTfidfIndexForChunks(flatChunks);
      setCorpusIdf(idf);

      setDocuments(builtDocs);
      const chunkCount = builtDocs.reduce((s, d) => s + d.chunks.length, 0);
      setLastSummary({ chunkCount, pageCount: totalPages });
      setProcessingStatus('Done');
      const totalChunks = chunkCount;
      const namesJoined = builtDocs.map((d) => d.name).join(', ');
      const welcomeText =
        builtDocs.length === 1
          ? `Document processed successfully! I've indexed **${totalChunks} chunks** from **${builtDocs[0].name}**. Ask me anything about the document — I'll only answer based on its content.`
          : `Document processed successfully! I've indexed **${totalChunks} chunks** from **${namesJoined}**. Ask me anything about these documents — I'll only answer based on their content.`;
      setChatHistory([
        {
          id: uuid(),
          role: 'assistant',
          content: welcomeText,
          timestamp: Date.now(),
        },
      ]);
      setPendingFiles([]);
      if (fileInputRef.current) fileInputRef.current.value = '';
      if (mergeFileInputRef.current) mergeFileInputRef.current.value = '';
      setView('chat');
    } catch (err) {
      setProcessingStatus(
        err instanceof Error ? err.message : 'Processing failed',
      );
    } finally {
      setIsProcessing(false);
    }
  }, [pendingFiles]);

  const mergePendingDocuments = useCallback(async () => {
    if (pendingFiles.length === 0) return;

    setView('processing');
    setIsProcessing(true);
    setProcessingStatus('Step 1/3: Starting…');

    try {
      const { builtDocs, totalPages } = await ingestFilesToBuiltDocs(
        pendingFiles,
        setProcessingStatus,
      );

      const merged = [...documents, ...builtDocs];
      const flatAll = merged.flatMap((d) => d.chunks);
      for (const ch of flatAll) {
        delete ch.tfidf;
      }
      const idf = buildTfidfIndexForChunks(flatAll);
      setCorpusIdf(idf);
      setDocuments(merged);
      const chunkCount = flatAll.length;
      setLastSummary((prev) => ({
        chunkCount,
        pageCount: prev.pageCount + totalPages,
      }));
      setProcessingStatus('Done');
      setChatHistory((h) => [
        ...h,
        {
          id: uuid(),
          role: 'assistant',
          content: `Added **${builtDocs.length}** document(s) and reindexed **${chunkCount}** total chunks. You can keep chatting — retrieval now covers the expanded library.`,
          timestamp: Date.now(),
        },
      ]);
      setPendingFiles([]);
      if (mergeFileInputRef.current) mergeFileInputRef.current.value = '';
      if (fileInputRef.current) fileInputRef.current.value = '';
      setView('chat');
    } catch (err) {
      setProcessingStatus(
        err instanceof Error ? err.message : 'Processing failed',
      );
    } finally {
      setIsProcessing(false);
    }
  }, [pendingFiles, documents]);

  const flatChunks = useMemo(
    () => documents.flatMap((d) => d.chunks),
    [documents],
  );

  useEffect(() => {
    const el = messageScrollRef.current;
    if (el) {
      el.scrollTo({ top: el.scrollHeight, behavior: 'smooth' });
    }
  }, [chatHistory, isAwaitingClaude, isThinking]);

  const isChatBusy = isAwaitingClaude || isThinking;
  const isInputLocked = isChatBusy || cooldownActive;

  const beginPostAnswerCooldown = useCallback(() => {
    const durationMs = getGeminiCooldownRemainingMs();
    if (durationMs <= 0) return;
    setCooldownDurationMs(durationMs);
    setCooldownActive(true);
    setCooldownFill(false);
  }, []);

  useEffect(() => {
    if (!cooldownActive || cooldownDurationMs <= 0) return undefined;
    const fillFrame = requestAnimationFrame(() => {
      requestAnimationFrame(() => setCooldownFill(true));
    });
    const doneTimer = setTimeout(() => {
      setCooldownActive(false);
      setCooldownFill(false);
      setInputReadyFlash(true);
    }, cooldownDurationMs);
    const flashOffTimer = setTimeout(() => {
      setInputReadyFlash(false);
    }, cooldownDurationMs + 1000);
    return () => {
      cancelAnimationFrame(fillFrame);
      clearTimeout(doneTimer);
      clearTimeout(flashOffTimer);
    };
  }, [cooldownActive, cooldownDurationMs]);

  const handleSend = useCallback(
    async (e) => {
      e.preventDefault();
      const q = queryInput.trim();
      if (!q || !corpusIdf || flatChunks.length === 0 || isChatBusy) {
        return;
      }
      if (sendLockRef.current) return;
      sendLockRef.current = true;

      try {
        const apiKey = import.meta.env.VITE_GEMINI_API_KEY;
        const userMsg = {
          id: uuid(),
          role: 'user',
          content: q,
          timestamp: Date.now(),
        };
        const afterUser = [...chatHistory, userMsg];
        setChatHistory(afterUser);
        setQueryInput('');

        const { retrievalQuery, isFollowUp } = resolveRetrievalQuery(
          q,
          chatHistory,
        );

        const { results, confidence } = retrieveRelevantChunks(
          retrievalQuery,
          flatChunks,
          7,
          corpusIdf,
        );

        const relevantForApi = results.map((r) => ({
          text: r.chunk.text,
          docName: r.chunk.docName,
          page: r.chunk.page,
        }));

        const displaySources = results.slice(0, 3).map((r) => ({
          docName: r.chunk.docName,
          page: r.chunk.page,
          preview:
            r.chunk.text.length > 150
              ? `${r.chunk.text.slice(0, 150)}…`
              : r.chunk.text,
          fullText: r.chunk.text,
          similarity: r.similarity,
        }));

        const tierFromConfidence =
          confidence.avg > 0.3
            ? 'high'
            : confidence.avg >= 0.1
              ? 'medium'
              : 'low';

        if (relevantForApi.length === 0) {
          setChatHistory([
            ...afterUser,
            {
              id: uuid(),
              role: 'assistant',
              content:
                "I couldn't find relevant information in the uploaded document(s) to answer this question. Please make sure your question relates to the document content.",
              timestamp: Date.now(),
              sources: [],
              confidenceTier: 'out_of_scope',
            },
          ]);
          return;
        }

        if (!apiKey) {
          setChatHistory([
            ...afterUser,
            {
              id: uuid(),
              role: 'assistant',
              content:
                'Missing API key. Create a `.env` file with `VITE_GEMINI_API_KEY=your_key` and restart the dev server.',
              timestamp: Date.now(),
              sources: displaySources,
              confidenceTier: tierFromConfidence,
              isError: true,
            },
          ]);
          return;
        }

        setPendingLabel('Thinking...');
        setIsThinking(true);
        let didCallGemini = false;
        try {
          const answerText = await answerQuestion(
            q,
            relevantForApi,
            afterUser,
            {
              onWait: (label) => {
                setPendingLabel(label);
                setIsThinking(true);
              },
              onWaitEnd: () => {
                setIsThinking(false);
              },
              onApiStart: () => {
                setIsThinking(false);
                setIsAwaitingClaude(true);
              },
            },
            { isFollowUp },
          );
          didCallGemini = true;
          setChatHistory((prev) => [
            ...prev,
            {
              id: uuid(),
              role: 'assistant',
              content: answerText,
              timestamp: Date.now(),
              sources: displaySources,
              confidenceTier: tierFromConfidence,
            },
          ]);
        } catch {
          didCallGemini = true;
          setChatHistory((prev) => [
            ...prev,
            {
              id: uuid(),
              role: 'assistant',
              content:
                "I couldn't reach the document service just now. Please try again in a moment.",
              timestamp: Date.now(),
              sources: displaySources,
              confidenceTier: tierFromConfidence,
            },
          ]);
        } finally {
          setIsAwaitingClaude(false);
          setIsThinking(false);
          if (didCallGemini) beginPostAnswerCooldown();
        }
      } finally {
        sendLockRef.current = false;
      }
    },
    [
      queryInput,
      corpusIdf,
      flatChunks,
      chatHistory,
      isChatBusy,
      beginPostAnswerCooldown,
    ],
  );

  const summaryLine = `Processed ${lastSummary.chunkCount} chunks from ${lastSummary.pageCount} pages`;

  const indexedStats = useMemo(
    () => ({
      docCount: documents.length,
      chunkCount: documents.reduce((n, d) => n + d.chunks.length, 0),
    }),
    [documents],
  );

  return (
    <>
      <DocqaKeyframes />
      <div className="flex min-h-screen flex-col bg-gray-950 text-gray-100">
        <header className="sticky top-0 z-40 flex h-12 shrink-0 items-center justify-between gap-2 border-b border-gray-800/90 bg-gray-950/90 px-3 backdrop-blur sm:px-4">
          <span className="text-base font-bold tracking-tight text-indigo-400">
            DocQA
          </span>
          <div className="flex flex-1 flex-wrap items-center justify-end gap-2 sm:gap-3">
            {view === 'chat' ? (
              <>
                <button
                  type="button"
                  onClick={clearChat}
                  className="rounded-md border border-gray-700 px-2.5 py-1 text-xs text-gray-200 hover:border-indigo-500/40 hover:text-white sm:px-3"
                >
                  Clear Chat
                </button>
                <button
                  type="button"
                  onClick={resetToUpload}
                  className="rounded-md border border-red-900/40 bg-red-950/30 px-2.5 py-1 text-xs text-red-200 hover:bg-red-950/50 sm:px-3"
                >
                  New Session
                </button>
              </>
            ) : null}
            <button
              type="button"
              onClick={() => setHelpOpen(true)}
              className="flex h-8 w-8 items-center justify-center rounded-full border border-gray-600 text-sm font-semibold text-gray-300 hover:border-indigo-500/50 hover:text-white"
              aria-label="How it works"
            >
              ?
            </button>
            <span className="max-w-[10rem] truncate rounded-full border border-indigo-500/40 bg-indigo-950/50 px-2 py-0.5 text-[9px] font-medium uppercase tracking-wide text-indigo-200 sm:max-w-none sm:text-[10px]">
              Powered by Gemini
            </span>
          </div>
        </header>

        <HelpModal open={helpOpen} onClose={() => setHelpOpen(false)} />

        {view !== 'chat' ? (
          <div className="relative flex min-h-0 flex-1 flex-col">
            <div className="pointer-events-none absolute inset-0 overflow-hidden">
              <div className="docqa-gradient-bg absolute -inset-[25%] bg-gradient-to-br from-violet-900/30 via-indigo-950/50 to-gray-950" />
            </div>
            <div className="relative mx-auto flex w-full max-w-2xl flex-1 flex-col gap-8 px-4 py-10">
          <header className="text-center">
            <h1 className="text-2xl font-semibold tracking-tight text-white">
              Context-Aware Document Q&A
            </h1>
            <p className="mt-2 text-sm text-gray-400">
              Upload documents, then chat with Gemini using TF-IDF retrieval
              over your files.
            </p>
            {view === 'upload' && uploadedFileCount > 0 && (
              <p className="mt-1 text-xs text-gray-600">
                {uploadedFileCount} file{uploadedFileCount !== 1 ? 's' : ''} in
                queue
              </p>
            )}
          </header>

          {view === 'upload' && (
            <section className="space-y-6">
              <div
                role="button"
                tabIndex={0}
                onKeyDown={(e) => {
                  if (e.key === 'Enter' || e.key === ' ') {
                    e.preventDefault();
                    fileInputRef.current?.click();
                  }
                }}
                onClick={() => fileInputRef.current?.click()}
                onDragEnter={onDragEnter}
                onDragLeave={onDragLeave}
                onDragOver={onDragOver}
                onDrop={onDrop}
                className="cursor-pointer rounded-2xl border-2 border-dashed border-indigo-500/40 bg-gray-900/50 px-8 py-14 text-center transition hover:border-indigo-500/70 hover:bg-gray-900"
              >
                <UploadIcon />
                <p className="mt-4 text-sm font-medium text-gray-200">
                  Drag & drop PDF or TXT files here
                </p>
                <p className="mt-1 text-xs text-gray-500">or click to browse</p>
                <input
                  ref={fileInputRef}
                  type="file"
                  accept=".pdf,.txt,application/pdf,text/plain"
                  multiple
                  className="hidden"
                  onChange={onInputChange}
                />
              </div>

              {pendingFiles.length > 0 && (
                <div className="rounded-xl border border-gray-800 bg-gray-900/80 p-4">
                  <p className="mb-3 text-xs font-medium uppercase tracking-wide text-gray-500">
                    Selected files ({pendingFiles.length})
                  </p>
                  <ul className="space-y-2">
                    {pendingFiles.map((f, idx) => (
                      <li
                        key={`${f.name}-${f.size}-${idx}`}
                        className="flex items-center justify-between gap-3 rounded-lg bg-gray-950/80 px-3 py-2 text-sm"
                      >
                        <span
                          className="truncate text-gray-200"
                          title={f.name}
                        >
                          {f.name}
                        </span>
                        <span className="shrink-0 text-xs text-gray-500">
                          {formatBytes(f.size)}
                        </span>
                        <button
                          type="button"
                          onClick={(e) => {
                            e.stopPropagation();
                            removeFileAt(idx);
                          }}
                          className="shrink-0 rounded-md px-2 py-1 text-xs text-indigo-400 hover:bg-indigo-500/10"
                        >
                          Remove
                        </button>
                      </li>
                    ))}
                  </ul>
                  <button
                    type="button"
                    onClick={processDocuments}
                    className="mt-4 w-full rounded-lg bg-indigo-500 py-3 text-sm font-medium text-white shadow-lg shadow-indigo-500/25 transition hover:bg-indigo-400"
                  >
                    Process Document
                  </button>
                </div>
              )}
            </section>
          )}

          {view === 'processing' && (
            <section className="flex flex-col items-center justify-center gap-6 py-20">
              <div
                className="h-12 w-12 animate-spin rounded-full border-2 border-gray-700 border-t-indigo-500"
                aria-hidden="true"
              />
              <p className="text-center text-sm text-gray-300">
                {processingStatus}
              </p>
              {!isProcessing &&
                processingStatus &&
                processingStatus !== 'Done' && (
                  <button
                    type="button"
                    onClick={resetToUpload}
                    className="text-sm text-indigo-400 hover:underline"
                  >
                    Back to upload
                  </button>
                )}
            </section>
          )}
            </div>
          </div>
      ) : (
        <div className="flex min-h-0 flex-1 flex-col lg:flex-row">
          <aside className="w-full shrink-0 border-b border-gray-800 bg-gray-900 lg:w-[30%] lg:border-b-0 lg:border-r lg:border-gray-800">
            <div className="sticky top-0 flex min-h-0 flex-col gap-4 p-4 lg:min-h-screen lg:overflow-y-auto">
              <div>
                <h1 className="text-lg font-semibold tracking-tight text-white">
                  Document Q&A
                </h1>
                <p className="mt-1 text-xs text-gray-500">{summaryLine}</p>
              </div>
              <div>
                <p className="mb-2 text-xs font-medium uppercase tracking-wide text-gray-500">
                  Indexed documents
                </p>
                <ul className="space-y-1">
                  {documents.map((d) => (
                    <li
                      key={d.id}
                      className="flex items-center justify-between gap-2 rounded-lg border border-gray-800/80 bg-gray-950/50 px-3 py-2 text-sm"
                    >
                      <span className="truncate text-gray-200" title={d.name}>
                        {d.name}
                      </span>
                      <span className="shrink-0 text-xs tabular-nums text-indigo-400">
                        {d.chunks.length} chunks
                      </span>
                    </li>
                  ))}
                </ul>
              </div>
              <input
                ref={mergeFileInputRef}
                type="file"
                accept=".pdf,.txt,application/pdf,text/plain"
                multiple
                className="hidden"
                onChange={(e) => {
                  addFiles(e.target.files);
                  e.target.value = '';
                }}
              />
              <button
                type="button"
                onClick={() => mergeFileInputRef.current?.click()}
                disabled={isProcessing}
                className="w-full rounded-lg border border-indigo-500/50 bg-indigo-950/40 py-2.5 text-sm text-indigo-200 transition hover:bg-indigo-900/50 disabled:opacity-40"
              >
                Add documents
              </button>
              {pendingFiles.length > 0 && view === 'chat' ? (
                <div className="rounded-lg border border-amber-500/35 bg-amber-950/25 p-3">
                  <p className="mb-2 text-[10px] font-semibold uppercase tracking-wide text-amber-200/90">
                    Staged for merge ({pendingFiles.length})
                  </p>
                  <ul className="mb-3 max-h-28 space-y-1 overflow-y-auto text-xs text-amber-100/90">
                    {pendingFiles.map((f, idx) => (
                      <li
                        key={`staged-${f.name}-${idx}`}
                        className="flex justify-between gap-2 truncate"
                      >
                        <span className="truncate">{f.name}</span>
                        <button
                          type="button"
                          className="shrink-0 text-amber-300 hover:text-white"
                          onClick={() => removeFileAt(idx)}
                        >
                          ×
                        </button>
                      </li>
                    ))}
                  </ul>
                  <button
                    type="button"
                    onClick={mergePendingDocuments}
                    disabled={isProcessing}
                    className="w-full rounded-lg bg-amber-600 py-2 text-xs font-medium text-white hover:bg-amber-500 disabled:opacity-40"
                  >
                    Merge & reindex
                  </button>
                </div>
              ) : null}
            </div>
          </aside>

          <main className="flex min-h-0 flex-1 flex-col lg:min-h-0 lg:w-[70%]">
            <div className="flex min-h-0 flex-1 flex-col border-gray-800 lg:border-l">
              <div className="shrink-0 border-b border-gray-800 bg-gray-950/80 px-3 py-2 text-center text-xs text-gray-400 sm:text-sm">
                <span className="inline-flex flex-wrap items-center justify-center gap-x-3 gap-y-1">
                  <span>📄 {indexedStats.docCount} docs</span>
                  <span className="text-gray-600">|</span>
                  <span>🧩 {indexedStats.chunkCount} chunks</span>
                  <span className="text-gray-600">|</span>
                  <span className="text-indigo-300">🔍 Ready</span>
                </span>
              </div>
              <div className="flex min-h-0 flex-1 flex-col px-3 py-3 sm:px-5">
                <div
                  ref={messageScrollRef}
                  className="min-h-0 flex-1 space-y-4 overflow-y-auto pb-4"
                >
                {chatHistory.map((msg, idx) => {
                  const key = msg.id || `msg-${idx}`;
                  const ts = formatMessageTime(msg.timestamp);
                  if (msg.role === 'user') {
                    return (
                      <div key={key} className="flex justify-end">
                        <div className="max-w-[min(100%,36rem)]">
                          <div className="rounded-tl-2xl rounded-bl-2xl rounded-br-sm bg-indigo-600 px-4 py-3 text-sm text-white shadow-md shadow-black/20">
                            <p className="whitespace-pre-wrap">{msg.content}</p>
                          </div>
                          {ts ? (
                            <p className="mt-1 text-right text-xs text-gray-500">
                              {ts}
                            </p>
                          ) : null}
                        </div>
                      </div>
                    );
                  }
                  return (
                    <div key={key} className="flex justify-start">
                      <div className="flex max-w-[min(100%,42rem)] gap-3">
                        <BotAvatar />
                        <div className="min-w-0 flex-1">
                          {msg.confidenceTier ? (
                            <div className="mb-1.5">
                              <ConfidenceBadge tier={msg.confidenceTier} />
                            </div>
                          ) : null}
                          <div
                            className={`rounded-tr-2xl rounded-br-2xl rounded-bl-2xl px-4 py-3 text-sm shadow-md shadow-black/15 ${
                              msg.isError
                                ? 'border border-red-900/50 bg-gray-900 text-red-100 ring-1 ring-red-900/40'
                                : 'bg-gray-800 text-gray-100'
                            }`}
                          >
                            <div className="whitespace-pre-wrap break-words">
                              <MessageRichText text={msg.content} />
                            </div>
                          </div>
                          <CollapsibleSources sources={msg.sources} />
                          {ts ? (
                            <p className="mt-2 text-xs text-gray-500">{ts}</p>
                          ) : null}
                        </div>
                      </div>
                    </div>
                  );
                })}

                {isChatBusy ? (
                  <div className="flex justify-start">
                    <div className="flex gap-3">
                      <BotAvatar />
                      <div className="rounded-tr-2xl rounded-br-2xl rounded-bl-2xl bg-gray-800 px-4 py-3">
                        <TypingIndicator
                          label={isThinking ? pendingLabel : undefined}
                        />
                      </div>
                    </div>
                  </div>
                ) : null}
              </div>

              <form
                onSubmit={handleSend}
                className="mt-auto shrink-0 border-t border-gray-800 pt-4"
              >
                <div className="flex gap-2">
                  <textarea
                    rows={2}
                    value={queryInput}
                    onChange={(e) => setQueryInput(e.target.value)}
                    onKeyDown={(e) => {
                      if (e.key === 'Enter' && !e.shiftKey) {
                        e.preventDefault();
                        if (
                          !isInputLocked &&
                          queryInput.trim() &&
                          corpusIdf &&
                          flatChunks.length > 0
                        ) {
                          handleSend(e);
                        }
                      }
                    }}
                    disabled={isInputLocked}
                    placeholder={
                      cooldownActive
                        ? 'Ready in a moment...'
                        : 'Ask a question about your document...'
                    }
                    className={`min-h-[3.25rem] min-w-0 flex-1 resize-y rounded-lg border bg-gray-950 px-4 py-3 text-sm text-white placeholder:text-gray-600 focus:outline-none focus:ring-1 disabled:opacity-50 ${
                      inputReadyFlash
                        ? 'border-green-500 ring-green-500/40 transition-colors duration-300'
                        : 'border-gray-700 focus:border-indigo-500 focus:ring-indigo-500'
                    }`}
                  />
                  <button
                    type="submit"
                    disabled={
                      isInputLocked ||
                      !corpusIdf ||
                      !queryInput.trim() ||
                      flatChunks.length === 0
                    }
                    className="h-fit shrink-0 self-end rounded-lg bg-indigo-500 px-5 py-3 text-sm font-medium text-white shadow-lg shadow-indigo-500/25 transition hover:bg-indigo-400 disabled:cursor-not-allowed disabled:opacity-40"
                  >
                    Send
                  </button>
                </div>
                {cooldownActive ? (
                  <div className="mt-3 space-y-2" aria-live="polite">
                    <p className="text-center text-xs font-medium tracking-wide text-gray-500">
                      Analyzing document for next question...
                    </p>
                    <div className="h-1 w-full overflow-hidden rounded-full bg-gray-800/90">
                      <div
                        className="h-full rounded-full bg-gradient-to-r from-indigo-600 via-indigo-500 to-blue-400"
                        style={{
                          width: cooldownFill ? '100%' : '0%',
                          transition: `width ${cooldownDurationMs}ms linear`,
                        }}
                      />
                    </div>
                  </div>
                ) : null}
              </form>
            </div>
            </div>
          </main>
        </div>
      )}
      </div>
    </>
  );
}
