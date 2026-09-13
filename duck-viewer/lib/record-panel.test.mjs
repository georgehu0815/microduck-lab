import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import vm from "node:vm";
import { test } from "node:test";
import ts from "typescript";

const source = readFileSync(
  new URL("../components/RecordPanel.tsx", import.meta.url),
  "utf8"
);
const { outputText } = ts.transpileModule(source, {
  compilerOptions: {
    jsx: ts.JsxEmit.ReactJSX,
    module: ts.ModuleKind.CommonJS,
    target: ts.ScriptTarget.ES2022,
    esModuleInterop: true,
  },
});

function childrenOf(node) {
  if (node == null || typeof node === "boolean") return [];
  if (Array.isArray(node)) return node.flatMap(childrenOf);
  if (typeof node !== "object") return [];
  return [node, ...childrenOf(node.props?.children)];
}

function textOf(node) {
  if (node == null || typeof node === "boolean") return "";
  if (Array.isArray(node)) return node.map(textOf).join("");
  if (typeof node !== "object") return String(node);
  return textOf(node.props?.children);
}

function button(tree, label) {
  return childrenOf(tree).find(
    (node) => node.type === "button" && node.props["aria-label"] === label
  );
}

function createHarness({ selected = null, phase = "idle" } = {}) {
  const events = [];
  const timeouts = [];
  const refs = [];
  const states = [];
  let hookIndex = 0;
  let currentSelected = selected;
  let capture = {
    phase,
    duckId: null,
    epoch: 0,
    recordingSince: phase === "recording" ? Date.now() - 2_000 : 0,
    result: null,
    error: phase === "error" ? "capture failed" : null,
  };

  const react = {
    useEffect(effect) {
      effect();
    },
    useRef(initial) {
      const index = hookIndex++;
      refs[index] ??= { current: initial };
      return refs[index];
    },
    useState(initial) {
      const index = hookIndex++;
      states[index] ??= typeof initial === "function" ? initial() : initial;
      return [states[index], (next) => {
        states[index] =
          typeof next === "function" ? next(states[index]) : next;
      }];
    },
  };

  const jsxRuntime = {
    Fragment: Symbol("Fragment"),
    jsx: (type, props) => ({ type, props: props ?? {} }),
    jsxs: (type, props) => ({ type, props: props ?? {} }),
  };

  class MockMediaRecorder {
    static isTypeSupported() {
      return true;
    }

    constructor(stream, options) {
      this.stream = stream;
      this.options = options;
      this.state = "inactive";
      events.push(["recorder-created", options.mimeType]);
    }

    start(timeslice) {
      this.state = "recording";
      events.push(["recorder-start", timeslice]);
    }

    stop() {
      this.state = "inactive";
      events.push(["recorder-stop"]);
      this.onstop?.();
    }

    emit(bytes = "frame") {
      this.ondataavailable?.({ data: new Blob([bytes]) });
    }
  }

  const canvas = {
    captureStream(fps) {
      events.push(["capture-stream", fps]);
      return {
        getVideoTracks: () => [{ requestFrame() {} }],
      };
    },
  };

  const record = {
    captureDone(result) {
      events.push(["capture-done", result.name]);
      capture = { ...capture, phase: "done", result };
    },
    captureError(error) {
      events.push(["capture-error", error]);
      capture = { ...capture, phase: "error", error };
    },
    captureFraming(duckId) {
      events.push(["capture-framing", duckId]);
      capture = {
        ...capture,
        phase: "framing",
        duckId,
        epoch: capture.epoch + 1,
        recordingSince: 0,
        result: null,
        error: null,
      };
    },
    captureProcessing() {
      events.push(["capture-processing"]);
      capture = { ...capture, phase: "processing", recordingSince: 0 };
    },
    captureRecording() {
      events.push(["capture-recording"]);
      capture = {
        ...capture,
        phase: "recording",
        recordingSince: Date.now(),
      };
    },
    captureReset() {
      events.push(["capture-reset"]);
      capture = {
        ...capture,
        phase: "idle",
        duckId: null,
        recordingSince: 0,
        result: null,
        error: null,
      };
    },
    getCapture: () => capture,
    getCaptureCanvas: () => canvas,
    getFramesPushed: () => 10,
    hasCaptureTrack: () => false,
    setCaptureTrack(track) {
      events.push(["capture-track", track === null ? "clear" : "set"]);
    },
    snapshotNow(name) {
      events.push(["snapshot", name]);
      return true;
    },
    useCapture: () => capture,
  };

  const dependencies = {
    react,
    "react/jsx-runtime": jsxRuntime,
    "@/lib/lab": { LAB_HTTP: "http://lab.test" },
    "@/lib/select": { useSelectedDuck: () => currentSelected },
    "@/lib/ui": { useHudRight: () => 0 },
    "@/lib/record": record,
    "./LanguageProvider": {
      useLanguage: () => ({
        language: "en",
        setLanguage() {},
        t: (english) => english,
      }),
    },
    "./Toasts": { pushToast: (message) => events.push(["toast", message]) },
  };
  const exports = {};
  const fetch = async (url, options) => {
    events.push(["fetch", url, options.method, options.body.size]);
    return {
      ok: true,
      json: async () => ({
        name: "duck-lab-take",
        mp4: "/captures/take.mp4",
        gif: "/captures/take.gif",
        mp4Kb: 2048,
        gifKb: 512,
        dir: "captures/duck-lab-take",
      }),
    };
  };

  vm.runInNewContext(
    outputText,
    {
      Blob,
      Date,
      MediaRecorder: MockMediaRecorder,
      clearInterval() {},
      clearTimeout() {},
      encodeURIComponent,
      exports,
      fetch,
      require(id) {
        assert.ok(id in dependencies, `Unexpected dependency: ${id}`);
        return dependencies[id];
      },
      setInterval() {
        return 1;
      },
      window: {
        setTimeout(callback, delay) {
          const timer = { callback, delay };
          timeouts.push(timer);
          return timeouts.length;
        },
      },
    },
    { filename: "RecordPanel.tsx" }
  );

  const clientRef = {
    current: {
      frame: {
        ducks: [
          { id: "duck-1", name: "Ada Duck" },
          { id: "duck-2", name: "Grace Duck" },
        ],
      },
    },
  };

  return {
    events,
    timeouts,
    get capture() {
      return capture;
    },
    get recorder() {
      return events
        .filter(([name]) => name === "recorder-created")
        .length
        ? refs[0].current
        : null;
    },
    render() {
      hookIndex = 0;
      return exports.RecordPanel({ clientRef });
    },
    setCapture(next) {
      capture = { ...capture, ...next };
    },
    setSelected(next) {
      currentSelected = next;
    },
  };
}

test("screenshot and video controls persist without a selected duck", () => {
  const harness = createHarness();
  const tree = harness.render();

  assert.ok(button(tree, "Take screenshot"));
  assert.equal(textOf(button(tree, "Take screenshot")), "📷 shot");
  assert.ok(button(tree, "Start video recording"));
  assert.equal(textOf(button(tree, "Start video recording")), "🎥 record");
});

test("starting without a selection records the current view with duckId null", () => {
  const harness = createHarness();
  const tree = harness.render();

  button(tree, "Start video recording").props.onClick();

  assert.equal(harness.capture.duckId, null);
  assert.equal(harness.capture.phase, "recording");
  assert.deepEqual(harness.timeouts, []);
  assert.deepEqual(
    harness.events.filter(([name]) =>
      ["capture-framing", "capture-stream", "recorder-start", "capture-recording"].includes(name)
    ),
    [
      ["capture-framing", null],
      ["capture-stream", 0],
      ["recorder-start", 250],
      ["capture-recording"],
    ]
  );
});

test("a selected duck keeps the framing delay and has no automatic stop timer", () => {
  const harness = createHarness({ selected: "duck-1" });
  const tree = harness.render();

  button(tree, "Start video recording").props.onClick();

  assert.equal(harness.capture.phase, "framing");
  assert.equal(harness.capture.duckId, "duck-1");
  assert.deepEqual(harness.timeouts.map(({ delay }) => delay), [1200]);
  assert.equal(
    harness.events.some(([name]) => name === "recorder-start"),
    false
  );

  harness.timeouts[0].callback();

  assert.equal(harness.capture.phase, "recording");
  assert.deepEqual(harness.timeouts.map(({ delay }) => delay), [1200]);
  assert.equal(
    harness.events.filter(([name]) => name === "recorder-start").length,
    1
  );
});

test("stop enters processing before uploading and completes the capture", async () => {
  const harness = createHarness();
  button(harness.render(), "Start video recording").props.onClick();
  harness.recorder.emit();

  button(harness.render(), "Stop recording").props.onClick();

  assert.equal(harness.capture.phase, "processing");
  const processingIndex = harness.events.findIndex(
    ([name]) => name === "capture-processing"
  );
  const stopIndex = harness.events.findIndex(([name]) => name === "recorder-stop");
  const fetchIndex = harness.events.findIndex(([name]) => name === "fetch");
  assert.ok(processingIndex < stopIndex);
  assert.ok(stopIndex < fetchIndex);
  assert.deepEqual(harness.events[fetchIndex].slice(0, 3), [
    "fetch",
    "http://lab.test/captures?name=duck-lab",
    "POST",
  ]);
  assert.ok(harness.events[fetchIndex][3] > 0);

  await new Promise((resolve) => setImmediate(resolve));

  assert.equal(harness.capture.phase, "done");
  assert.equal(harness.capture.result.name, "duck-lab-take");
});

test("capture controls remain visible through processing, done, and error and can restart", () => {
  for (const phase of ["processing", "done", "error"]) {
    const harness = createHarness({ phase });
    if (phase === "done") {
      harness.setCapture({
        result: {
          name: "take",
          mp4: "/take.mp4",
          gif: "/take.gif",
          mp4Kb: 1,
          gifKb: 1,
          dir: "captures/take",
        },
      });
    }
    const tree = harness.render();
    const shot = button(tree, "Take screenshot");
    const video = button(tree, "Start video recording");

    assert.ok(shot, `screenshot button missing during ${phase}`);
    assert.ok(video, `video button missing during ${phase}`);
    assert.equal(video.props.disabled, phase === "processing");

    if (phase !== "processing") {
      video.props.onClick();
      assert.equal(harness.capture.phase, "recording");
      assert.equal(harness.capture.duckId, null);
    }
  }
});
