// Step 4: a JavaScript REPL whose only API is a `computer` global. The agent writes a few lines of code per call
// (batching actions, looping over rows, checking results) instead of one tool call per action.
// JavaScriptCore gives a sandbox with no file system, network or process: only what is exposed here.
import Foundation
import JavaScriptCore

final class Repl {
    let context: JSContext
    let call: (String, JSON) async -> (content: [Content], isError: Bool)
    private var output: [String] = []
    private var images: [Content] = []

    init(call: @escaping (String, JSON) async -> (content: [Content], isError: Bool)) {
        self.call = call
        context = JSContext()!
        let printer: @convention(block) (String) -> Void = { [weak self] s in self?.output.append(s) }
        context.setObject(printer, forKeyedSubscript: "__print" as NSString)
        let bridge: @convention(block) (String, String) -> [String: Any] = { [weak self] tool, argsJSON in
            self?.bridge(tool, argsJSON) ?? ["text": "driver gone", "isError": true]
        }
        context.setObject(bridge, forKeyedSubscript: "__call" as NSString)
        context.evaluateScript(Repl.prelude)
    }

    /// Runs one native tool call to completion from inside JavaScript (which waits for it).
    private func bridge(_ tool: String, _ argsJSON: String) -> [String: Any] {
        let args = (try? JSONSerialization.jsonObject(with: Data(argsJSON.utf8))) as? JSON ?? [:]
        let done = DispatchSemaphore(value: 0)
        nonisolated(unsafe) var result: (content: [Content], isError: Bool) = ([], true)
        let call = self.call
        Task.detached {
            result = await call(tool, args)
            done.signal()
        }
        done.wait()
        var texts: [String] = []
        for c in result.content {
            switch c {
            case .text(let t): texts.append(t)
            case .image: images.append(c)
            }
        }
        return ["text": texts.joined(separator: "\n"), "isError": result.isError]
    }

    /// Runs a cell. Top-level `const/let/var x =` lines become globals, so names persist between calls.
    func run(code: String) -> (content: [Content], isError: Bool) {
        output = []
        images = []
        let lines = code.components(separatedBy: "\n").map { line -> String in
            guard let m = line.range(of: #"^(const|let|var)\s+([A-Za-z_$][\w$]*)\s*="#, options: .regularExpression)
            else { return line }
            let name = line[m].replacingOccurrences(of: #"^(const|let|var)\s+"#, with: "", options: .regularExpression)
                .replacingOccurrences(of: #"\s*=$"#, with: "", options: .regularExpression)
            return "globalThis.\(name) =" + line[m.upperBound...]
        }
        let wrapped = "globalThis.__result = undefined; globalThis.__error = undefined;\n"
            + "(async () => {\n" + lines.joined(separator: "\n") + "\n})().then(v => { globalThis.__result = v; },"
            + " e => { globalThis.__error = (e && e.stack) ? String(e) + '\\n' + e.stack.split('\\n').slice(0, 3).join('\\n') : String(e); });"
        context.exception = nil
        context.evaluateScript(wrapped)
        var isError = false
        if let ex = context.exception, !ex.isUndefined {
            output.append("SyntaxError or exception: \(ex)")
            isError = true
        }
        if let err = context.objectForKeyedSubscript("__error"), !err.isUndefined, !err.isNull {
            output.append("Error: \(err)")
            isError = true
        } else if let v = context.objectForKeyedSubscript("__result"), !v.isUndefined, !v.isNull {
            output.append(v.isObject ? (v.toObject().map { "\($0)" } ?? "\(v)") : "\(v)")
        }
        let text = output.isEmpty ? "(no output)" : output.joined(separator: "\n")
        return ([.text(String(text.prefix(60_000)))] + images, isError)
    }

    static let prelude = #"""
    globalThis.console = { log: (...a) => __print(a.map(x => typeof x === 'string' ? x : JSON.stringify(x)).join(' ')) };
    console.error = console.log; console.info = console.log; console.warn = console.log;
    function __invoke(tool, args) {
      const r = __call(tool, JSON.stringify(args));
      if (r.isError) throw new Error(r.text);
      return r.text;
    }
    function __do(tool, args) { const t = __invoke(tool, args); __print(t); return t; }
    function __target(t) {
      if (Array.isArray(t)) return { x: t[0], y: t[1] };
      if (typeof t === 'number') return { ref: t };
      throw new Error('a target is an element id N or a point [x, y]');
    }
    globalThis.sleep = (ms) => { __invoke('wait', { seconds: Math.min(10, (ms || 0) / 1000) }); };
    class App {
      constructor(name) { this.name = name; }
      getAXState(opts = {}) { const t = __invoke('get_state', Object.assign({ app: this.name }, opts)); if (opts.emit !== false) __print(t); return t; }
      getScreenshot() { const t = __invoke('screenshot', { app: this.name }); __print(t); return t; }
      click(target, opts = {}) { return __do('click', Object.assign({ app: this.name, button: opts.mouseButton, count: opts.clickCount, modifiers: opts.modifiers }, __target(target))); }
      setValue(id, value) { return __do('set_value', { app: this.name, ref: id, value: String(value) }); }
      typeText(text, id) { return __do('type_text', Object.assign({ app: this.name, text: String(text) }, id === undefined ? {} : { ref: id })); }
      pressKey(keys) { return __do('press_key', { app: this.name, keys }); }
      scroll(target, direction = 'down', amount = 3) { return __do('scroll', Object.assign({ app: this.name, direction, amount }, __target(target))); }
      drag(from, to) { const a = __target(from), b = __target(to);
        return __do('drag', { app: this.name, from_ref: a.ref, from_x: a.x, from_y: a.y, to_ref: b.ref, to_x: b.x, to_y: b.y }); }
      hover(target) { return __invoke('hover', Object.assign({ app: this.name }, __target(target))); }
      bringToFront() { return __invoke('bring_to_front', { app: this.name }); }
    }
    globalThis.computer = {
      listApps(opts = {}) { const t = __invoke('list_apps', {}); if (opts.emit !== false) __print(t); return t; },
      getApp(name, opts = {}) { const app = new App(name); app.getAXState(Object.assign({ full: true }, opts)); return app; },
    };
    """#
}
