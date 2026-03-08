import { useState } from "react";
import "./App.css";
import {
  NewspaperIcon,
  ExclamationTriangleIcon,
  CheckBadgeIcon,
  GlobeAmericasIcon,
} from "@heroicons/react/24/outline";

// URL your FastAPI backend exposes
const API_URL = "https://yunatakele-fake-news-detector-api.hf.space/predict";

function App() {
  const [text, setText] = useState("");
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [mousePos, setMousePos] = useState({ x: 0, y: 0 });

  const handlePredict = async () => {
    if (!text.trim()) return;

    setLoading(true);
    setResult(null);

    try {
      const response = await fetch(API_URL, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({ text }),
      });

      if (!response.ok) {
        throw new Error("Backend error");
      }

      const data = await response.json();
      setResult(data);
    } catch (error) {
      console.error(error);
      setResult({
        label: "ERROR",
        confidence: 0,
        explanation: "Could not connect to backend.",
      });
    }

    setLoading(false);
  };

  const handleMouseMove = (e) => {
    setMousePos({ x: e.clientX, y: e.clientY });
  };

  const isFake = result && result.label === "FAKE";
  const isReal = result && result.label === "REAL";

  const confidenceValue =
    result && typeof result.confidence === "number"
      ? Math.max(0, Math.min(1, result.confidence))
      : null;

  const confidencePercent =
    confidenceValue !== null ? Math.round(confidenceValue * 100) : null;

  const parallaxStyleLeft = {
    transform: `translate3d(${mousePos.x * 0.02}px, ${mousePos.y * 0.01}px, 0)`,
  };

  const parallaxStyleRight = {
    transform: `translate3d(${mousePos.x * -0.015}px, ${mousePos.y * -0.008}px, 0)`,
  };

  return (
    <div
      className="min-h-screen w-full bg-slate-950 text-slate-50 relative overflow-hidden"
      onMouseMove={handleMouseMove}
    >
      {/* Background newsroom-ish glow and parallax blobs */}
      <div
        className="pointer-events-none absolute -top-56 -left-40 h-96 w-96 rounded-full bg-blue-500/25 blur-3xl transition-transform duration-300"
        style={parallaxStyleLeft}
      />
      <div
        className="pointer-events-none absolute -bottom-64 -right-40 h-[26rem] w-[26rem] rounded-full bg-amber-500/20 blur-3xl transition-transform duration-300"
        style={parallaxStyleRight}
      />
      <div className="pointer-events-none absolute inset-0 bg-[radial-gradient(circle_at_top,_rgba(148,163,184,0.16),_transparent_60%)]" />

      {/* Subtle floating particles */}
      <div className="pointer-events-none absolute inset-0 overflow-hidden">
        <span className="particle-dot top-24 left-1/4" />
        <span className="particle-dot top-40 right-1/5" />
        <span className="particle-dot bottom-32 left-1/3" />
        <span className="particle-dot bottom-16 right-1/4" />
      </div>

      {/* Content layer */}
      <div className="relative z-10 flex flex-col min-h-screen">
        {/* HEADER */}
        <header className="w-full border-b border-slate-800/70 bg-slate-950/70 backdrop-blur">
          <div className="w-full px-6 py-4 flex items-center justify-between gap-3">
            <div className="flex items-center gap-2">
              <div className="flex h-9 w-9 items-center justify-center rounded-lg bg-blue-500/20 border border-blue-400/40 shadow-[0_0_20px_rgba(59,130,246,0.35)]">
                <NewspaperIcon className="h-5 w-5 text-blue-300" />
              </div>
              <div className="leading-tight">
                <p className="text-sm font-semibold text-slate-50">
                  Fake News Detector
                </p>
              </div>
            </div>
          </div>
        </header>

        {/* MAIN */}
        <main className="w-full px-4 sm:px-6 py-8 sm:py-10 flex-1">
          {/* Hero section with spotlight + waveform */}
          <section className="relative mb-10 text-center">
            {/* spotlight */}
            <div className="pointer-events-none absolute -top-24 left-1/2 -translate-x-1/2 h-48 w-[60%] bg-gradient-to-b from-blue-500/30 via-transparent to-transparent blur-3xl animate-spotlight" />

            <h1 className="text-4xl sm:text-5xl font-semibold tracking-tight text-slate-50">
              Fake News Detector
            </h1>

            <p className="mt-3 text-sm sm:text-base text-slate-400 max-w-2xl mx-auto">
              Paste a short news snippet below and get a model-based classification.
            </p>
            
          </section>


          {/* Main two-column layout */}
          <div className="mt-4 grid gap-8 lg:grid-cols-[minmax(0,1.8fr)_minmax(0,1.2fr)] w-full">
            {/* LEFT: Input panel */}
            <section className="bg-slate-900/80 border border-slate-800/80 rounded-2xl sm:rounded-3xl shadow-[0_24px_70px_rgba(15,23,42,0.9)] p-5 sm:p-6 space-y-4 backdrop-blur-md">
              <div className="flex items-center justify-between gap-2">
                <div className="text-left">
                  <p className="text-sm font-medium text-slate-100">
                    Paste content to investigate
                  </p>
                </div>
                <span className="inline-flex items-center rounded-full border border-slate-700/80 bg-slate-950 px-3 py-1 text-[11px] text-slate-400">
                  Press &quot;Check&quot; to run analysis
                </span>
              </div>

              <textarea
                className="w-full max-w-full p-3.5 sm:p-4 border border-slate-800/80 rounded-2xl bg-slate-950/70 shadow-inner focus:outline-none focus:ring-2 focus:ring-blue-500/80 focus:border-blue-500/70 text-sm sm:text-[15px] resize-none placeholder:text-slate-600"
                rows="7"
                value={text}
                onChange={(e) => setText(e.target.value)}
                placeholder="Example: A shocking new study claims that drinking coffee three times a day can extend human life to 150 years..."
              ></textarea>

              <div className="flex flex-col sm:flex-row items-center justify-between gap-3">
                <button
                  onClick={handlePredict}
                  disabled={loading || !text.trim()}
                  className={`inline-flex items-center justify-center px-6 py-2.5 rounded-full text-sm font-medium transition shadow-lg shadow-blue-500/25
                    ${
                      loading || !text.trim()
                        ? "bg-slate-700 text-slate-400 cursor-not-allowed shadow-none"
                        : "bg-blue-500 hover:bg-blue-600 text-slate-50"
                    }`}
                >
                  {loading ? "Analyzing…" : "Check this story"}
                </button>

                {loading && (
                  <div className="flex items-center gap-2 text-xs text-slate-400">
                    <span className="inline-flex h-2.5 w-2.5 rounded-full bg-blue-400 animate-ping" />
                    <span className="typing-dots">
                      <span>.</span>
                      <span>.</span>
                      <span>.</span>
                    </span>
                    <span>Scanning language patterns…</span>
                  </div>
                )}
              </div>
            </section>

            {/* RIGHT: Verdict panel */}
            <aside className="space-y-4">
              <div className="bg-slate-900/80 border border-slate-800/80 rounded-2xl sm:rounded-3xl p-5 sm:p-6 shadow-[0_24px_60px_rgba(15,23,42,0.96)] space-y-4 backdrop-blur-md">
                <div className="flex items-center justify-between gap-3">
                  <div>
                    <p className="text-xs uppercase tracking-[0.2em] text-slate-500">
                      Result
                    </p>
                    <p className="text-sm font-medium text-slate-100">
                      Model Output
                    </p>
                  </div>
                </div>

                {!result && !loading && (
                  <div className="mt-2 rounded-xl border border-dashed border-slate-700/80 bg-slate-950/40 px-4 py-6 text-sm text-slate-500">
                    <p className="font-medium text-slate-300 mb-1">
                      No analysis yet
                    </p>
                    <p>
                      Paste a news snippet on the left and click{" "}
                      <span className="font-semibold text-slate-200">
                        &quot;Analyze Text&quot;
                      </span>{" "}
                      to see how the model classifies it.
                    </p>
                  </div>
                )}

                {result && (
                  <div className="space-y-4">
                    {/* Top row: main verdict + badge */}
                    <div className="flex items-center gap-3">
                      <div
                        className={`flex h-11 w-11 items-center justify-center rounded-full border 
                          ${
                            isFake
                              ? "bg-red-900/40 border-red-500/60 text-red-300"
                              : isReal
                              ? "bg-emerald-900/40 border-emerald-500/60 text-emerald-300"
                              : "bg-amber-900/40 border-amber-500/60 text-amber-300"
                          }`}
                      >
                        {isFake ? (
                          <ExclamationTriangleIcon className="h-6 w-6" />
                        ) : (
                          <CheckBadgeIcon className="h-6 w-6" />
                        )}
                      </div>
                      <div>
                        <p className="text-xs uppercase tracking-[0.18em] text-slate-500">
                          Model assessment
                        </p>
                        <p className="text-sm font-semibold text-slate-50">
                          {isFake
                            ? "Potentially misleading or fake"
                            : isReal
                            ? "Likely credible in style"
                            : result.label}
                        </p>
                      </div>
                    </div>

                    {/* Chips row */}
                    <div className="flex flex-wrap items-center gap-2 text-xs">
                      <span className="inline-flex items-center rounded-full bg-slate-900 px-3 py-1 border border-slate-700 text-slate-100">
                        Label: {result.label}
                      </span>
                      {confidencePercent !== null && (
                        <span className="inline-flex items-center rounded-full bg-slate-900 px-3 py-1 border border-slate-700 text-slate-200">
                          Confidence: {confidencePercent}%
                        </span>
                      )}
                    </div>

                    {/* Confidence bar */}
                    {confidencePercent !== null && (
                      <div className="space-y-1">
                        <div className="flex justify-between text-[11px] text-slate-500">
                          <span>Confidence level</span>
                          <span>{confidencePercent}%</span>
                        </div>
                        <div className="h-2 rounded-full bg-slate-800 overflow-hidden">
                          <div
                            className={`h-full rounded-full transition-all duration-500 ${
                              isFake
                                ? "bg-gradient-to-r from-red-500 to-orange-400"
                                : isReal
                                ? "bg-gradient-to-r from-emerald-400 to-blue-400"
                                : "bg-gradient-to-r from-amber-400 to-yellow-300"
                            }`}
                            style={{ width: `${confidencePercent}%` }}
                          />
                        </div>
                      </div>
                    )}

                    {/* Explanation */}
                    <div className="rounded-xl bg-slate-950/60 border border-slate-800/80 p-3.5 text-sm text-slate-200">
                      <p className="font-medium text-slate-100 mb-1">
                        Explanation
                      </p>
                      <p>{result.explanation}</p>
                    </div>

                    <p className="text-[11px] text-slate-500">
                      This model looks at linguistic patterns similar to those
                      seen in fake vs. real news datasets. It does{" "}
                      <span className="font-semibold text-slate-300">not</span>{" "}
                      verify facts against external databases or news wires.
                    </p>
                  </div>
                )}
              </div>
            </aside>
          </div>
        </main>

        {/* FOOTER */}
        <footer className="w-full border-t border-slate-900/80 bg-slate-950/80 backdrop-blur-sm">
          <div className="w-full px-6 py-4 text-xs text-slate-500 flex flex-col sm:flex-row items-center justify-between gap-2">
            <p>Built by Yuna • Fake News Detector Project</p>
            <p className="text-[11px] text-slate-600">
              For educational use only · Always cross-check with trusted
              sources.
            </p>
          </div>
        </footer>
      </div>
    </div>
  );
}

export default App;
