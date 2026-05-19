# Parakeet ASR — Web client

React 18 + Vite + TypeScript SPA that talks to the FastAPI backend in `app/main.py`. The build output is dropped into `../static/`, which the backend already mounts as `/static` and serves at `/`.

## Quick start

```bash
npm install
npm run dev       # http://localhost:5173, proxies /v1, /health, /readyz to :8777
npm run build     # → ../static (the backend serves this in prod)
npm run lint      # eslint
```

Point the dev proxy at a non-local backend with:

```bash
PARAKEET_URL=https://gpu-box.example:8443 npm run dev
```

## Project layout

```
src/
  main.tsx, App.tsx        Entry + top-level layout
  App.css                  Design tokens (light/dark), component styles
  lib/
    api.ts                 REST + WS clients. postTranscription(),
                           connectLiveWS(). One source of truth for URLs.
    types.ts               TS interfaces mirroring the backend response shape.
                           Whisper segment / Word / WSConfig / WSMessage union.
    settings.ts            Zustand store + `persist` middleware (localStorage).
                           All user-tunable knobs (response_format, strategy,
                           timestamp_granularities, live_latency, etc.).
    mic.ts                 useMic() hook — getUserMedia + MediaRecorder
                           (webm/opus) lifecycle + AnalyserNode level meter.
    playback.ts            Singleton <audio> element, useSyncExternalStore-
                           backed time hook, setAudioFile() / seek() helpers.
  components/
    FileUploadPanel.tsx    Drag-drop + Transcribe (REST).
    LiveCapturePanel.tsx   Record button, level meter, drives mic + WS.
    SettingsPanel.tsx      All knobs, inline help text.
    OutputView.tsx         Dispatches on result.format to a subview.
    outputs/
      JsonView.tsx         response_format=json (compact).
      VerboseJsonView.tsx  Segments table + words + Whisper meta.
      TextView.tsx         Plain text.
      SrtView.tsx          srt / vtt (one component, variant prop).
    SegmentList.tsx        Whisper-style segments table. Click → seek + expand.
    WordTimeline.tsx       Word chips, active word highlight, click → seek.
    ThemeToggle.tsx        System / Light / Dark.
```

## Adding a new component

1. Create the file in `src/components/`. Export a named function component with a typed `Props` interface; do not default-export.
2. If it needs persisted state, add the field to `Settings` in `src/lib/settings.ts` (plus a default in `DEFAULTS`) and `useSettings()` it directly — do not pass settings through props.
3. If it consumes server data, use the types in `src/lib/types.ts`. If a new server message shape appears, add it to the discriminated union there; do not cast in component code.
4. Styles live in `src/App.css` under a scoped class prefix (`.foo`, `.foo__bar`). Use the CSS variables from `:root` so the component reacts to theme changes.

## Backend touchpoints

- REST: `POST /v1/audio/transcriptions` — multipart form, server-side query params for extensions. See `lib/api.ts:postTranscription`.
- WS: `WS /v1/audio/transcriptions` — JSON config first frame, binary audio chunks, empty binary frame for EOF. See `lib/api.ts:connectLiveWS`.
- Live mic format: `audio/webm;codecs=opus`, declared to the server as `format: "webm"`. ffmpeg in the backend image auto-detects from the bitstream.

The TypeScript types in `lib/types.ts` are the source of truth on the client. Keep them in sync with `app/streaming_v2.py:_whisper_segment`, `tokens_to_words`, and the WS payload constructors in `app/main.py`.

## Build target

`vite.config.ts` sets `build.outDir = '../static'` and `base = '/static/'`. The latter rewrites asset URLs in `index.html` to `/static/assets/...`, matching the FastAPI `StaticFiles` mount in `app/main.py`. Production builds work without backend changes.
