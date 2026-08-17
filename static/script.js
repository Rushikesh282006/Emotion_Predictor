(() => {
  'use strict';

  const EMOTION_META = {
    sadness:  { emoji: '😞', color: '--c-sadness'  },
    joy:      { emoji: '😆', color: '--c-joy'      },
    love:     { emoji: '😍', color: '--c-love'     },
    anger:    { emoji: '😡', color: '--c-anger'    },
    fear:     { emoji: '😱', color: '--c-fear'     },
    surprise: { emoji: '😮', color: '--c-surprise' },
  };

  const EMOTION_ORDER = ['sadness', 'joy', 'love', 'anger', 'fear', 'surprise'];

  const el = {
    statusDot: document.getElementById('statusDot'),
    statusText: document.getElementById('statusText'),
    textInput: document.getElementById('textInput'),
    charCount: document.getElementById('charCount'),
    scanBtn: document.getElementById('scanBtn'),
    inputFrame: document.getElementById('inputFrame'),
    scanSweep: document.getElementById('scanSweep'),
    readout: document.getElementById('readout'),
    idleState: document.getElementById('idleState'),
    errorState: document.getElementById('errorState'),
    errorMessage: document.getElementById('errorMessage'),
    resultState: document.getElementById('resultState'),
    resultEmoji: document.getElementById('resultEmoji'),
    resultLabel: document.getElementById('resultLabel'),
    resultConfidence: document.getElementById('resultConfidence'),
    spectrum: document.getElementById('spectrum'),
  };

  const root = document.documentElement;
  const cssVar = (name) => getComputedStyle(root).getPropertyValue(name).trim();

  let modelReady = false;
  let healthTimer = null;

  // ---------------------------------------------------------------
  // Health check
  // ---------------------------------------------------------------

  async function checkHealth() {
    try {
      const res = await fetch('/health', { cache: 'no-store' });
      if (!res.ok) throw new Error('bad status');
      const data = await res.json();

      if (data.model_loaded) {
        modelReady = true;
        el.statusDot.className = 'status-dot is-ready';
        el.statusText.textContent = 'model ready';
        clearInterval(healthTimer);
      } else {
        modelReady = false;
        el.statusDot.className = 'status-dot is-loading';
        el.statusText.textContent = 'loading model…';
      }
    } catch (err) {
      modelReady = false;
      el.statusDot.className = 'status-dot is-down';
      el.statusText.textContent = 'server unreachable';
    }
    updateScanBtnState();
  }

  function startHealthPolling() {
    checkHealth();
    healthTimer = setInterval(checkHealth, 4000);
  }

  // ---------------------------------------------------------------
  // Input handling
  // ---------------------------------------------------------------

  function updateCharCount() {
    const len = el.textInput.value.length;
    el.charCount.textContent = `${len} / 2000`;
  }

  function updateScanBtnState() {
    const hasText = el.textInput.value.trim().length > 0;
    el.scanBtn.disabled = !hasText || !modelReady;
  }

  el.textInput.addEventListener('input', () => {
    updateCharCount();
    updateScanBtnState();
  });

  el.textInput.addEventListener('keydown', (e) => {
    const isSubmitCombo = (e.metaKey || e.ctrlKey) && e.key === 'Enter';
    if (isSubmitCombo && !el.scanBtn.disabled) {
      e.preventDefault();
      runScan();
    }
  });

  el.scanBtn.addEventListener('click', runScan);

  // ---------------------------------------------------------------
  // Scan flow
  // ---------------------------------------------------------------

  function setBusy(isBusy) {
    el.scanBtn.disabled = isBusy || !modelReady || el.textInput.value.trim().length === 0;
    el.scanBtn.classList.toggle('is-busy', isBusy);
    el.scanBtn.querySelector('.scan-btn__label').textContent = isBusy ? 'Scanning…' : 'Scan text';
    el.inputFrame.classList.toggle('is-scanning', isBusy);
  }

  function showState(state) {
    el.readout.dataset.state = state;
  }

  function showError(message) {
    el.errorMessage.textContent = message;
    showState('error');
  }

  async function runScan() {
    const text = el.textInput.value.trim();
    if (!text || !modelReady) return;

    setBusy(true);

    try {
      const res = await fetch('/predict', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ text }),
      });

      if (res.status === 503) {
        showError("Signal lost — the model hasn't finished loading. Try again in a few seconds.");
        return;
      }

      if (!res.ok) {
        let detail = 'The scan failed. Try a different sentence.';
        try {
          const errBody = await res.json();
          if (errBody && errBody.detail) {
            detail = typeof errBody.detail === 'string'
              ? errBody.detail
              : 'The scan failed. Try a different sentence.';
          }
        } catch (_) { /* ignore parse failure, use default */ }
        showError(detail);
        return;
      }

      const data = await res.json();
      renderResult(data);
    } catch (err) {
      showError("Couldn't reach the scanner. Check your connection and try again.");
    } finally {
      setBusy(false);
    }
  }

  // ---------------------------------------------------------------
  // Rendering
  // ---------------------------------------------------------------

  function renderResult(data) {
    const meta = EMOTION_META[data.predicted_emotion] || { emoji: '❔', color: '--text' };
    const color = cssVar(meta.color);

    el.resultEmoji.textContent = meta.emoji;
    el.resultEmoji.style.setProperty('--emotion-color', color);
    el.resultLabel.textContent = data.predicted_emotion;
    el.resultLabel.style.setProperty('--emotion-color', color);
    el.resultConfidence.textContent = Math.round(data.confidence * 100);

    buildSpectrum(data.all_prob, data.predicted_emotion);

    // retrigger pop-in animation on the emoji
    el.resultEmoji.style.animation = 'none';
    void el.resultEmoji.offsetWidth;
    el.resultEmoji.style.animation = '';

    showState('result');
  }

  function buildSpectrum(allProb, topEmotion) {
    el.spectrum.innerHTML = '';

    EMOTION_ORDER.forEach((key, i) => {
      const prob = allProb[key] ?? 0;
      const pct = Math.round(prob * 100);
      const meta = EMOTION_META[key];
      const color = cssVar(meta.color);

      const bar = document.createElement('div');
      bar.className = 'bar' + (key === topEmotion ? ' is-top' : '');
      bar.style.setProperty('--bar-color', color);

      bar.innerHTML = `
        <div class="bar__track">
          <div class="bar__fill" style="height:0%"></div>
        </div>
        <span class="bar__emoji" aria-hidden="true">${meta.emoji}</span>
        <span class="bar__label">${key}</span>
        <span class="bar__pct">${pct}%</span>
      `;

      el.spectrum.appendChild(bar);

      const fill = bar.querySelector('.bar__fill');
      const delay = 60 + i * 55;
      setTimeout(() => {
        fill.style.height = `${Math.max(pct, 2)}%`;
      }, delay);
    });
  }

  // ---------------------------------------------------------------
  // Init
  // ---------------------------------------------------------------

  updateCharCount();
  updateScanBtnState();
  startHealthPolling();
})();