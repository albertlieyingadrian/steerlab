// ourdriver: a computer-use driver for macOS, served as MCP tools over stdio.
// Step 3 of the build ladder: step 2 plus background control (accessibility actions first, then events posted
// to the app's process); bring_to_front and hover only when asked.
import AppKit
import ApplicationServices

let VERSION = "0.5.1-step5"
/// Step 4 serves one tool, `repl`, in place of the separate tools (which the REPL calls inside).
let REPL_MODE = true

final class Driver {
    var states: [pid_t: AppState] = [:]
    lazy var repl = Repl(call: { [unowned self] name, args in await self.call(name, args) })

    func state(_ app: NSRunningApplication) -> AppState {
        if let s = states[app.processIdentifier] { return s }
        let s = AppState(pid: app.processIdentifier)
        states[app.processIdentifier] = s
        return s
    }

    // MARK: tools

    func listApps() -> String {
        var lines: [String] = []
        for app in AX.runningApps() {
            let pid = app.processIdentifier
            let name = app.localizedName ?? "?"
            let wins = AX.windows(of: pid)
            let titles = wins.enumerated().map { i, w -> String in
                let t = AX.string(w, kAXTitleAttribute) ?? ""
                let f = AX.frame(w).map { " (\(Int($0.width))x\(Int($0.height)))" } ?? ""
                return "    window \(i): \"\(t)\"\(f)"
            }
            lines.append("\(name) (pid \(pid)\(app.isActive ? ", frontmost" : ""))")
            lines.append(contentsOf: titles)
        }
        return lines.joined(separator: "\n")
    }

    func pickWindow(_ app: NSRunningApplication, _ which: Any?) throws -> AXUIElement {
        let wins = AX.windows(of: app.processIdentifier)
        guard !wins.isEmpty else { throw ToolError(message: "\(app.localizedName ?? "The app") has no open window") }
        if let i = which as? Int {
            guard wins.indices.contains(i) else { throw ToolError(message: "No window \(i); it has \(wins.count)") }
            return wins[i]
        }
        if let s = which as? String, !s.isEmpty {
            if let w = wins.first(where: { (AX.string($0, kAXTitleAttribute) ?? "").localizedCaseInsensitiveContains(s) }) {
                return w
            }
            throw ToolError(message: "No window titled like \"\(s)\"")
        }
        if let focused = AX.attr(AX.app(pid: app.processIdentifier), kAXFocusedWindowAttribute) {
            return focused as! AXUIElement
        }
        return wins[0]
    }

    func getState(_ args: JSON) async throws -> [Content] {
        let app = try AX.findApp(try str(args, "app"))
        let window = try pickWindow(app, args["window"])
        guard let wf = AX.frame(window) else { throw ToolError(message: "The window has no frame") }
        let st = state(app)
        let body = st.snapshot(window: window, full: (args["full"] as? Bool) ?? false, query: args["query"] as? String,
                               frames: (args["frames"] as? Bool) ?? false)
        let title = AX.string(window, kAXTitleAttribute) ?? ""
        var out: [Content] = [.text("\(app.localizedName ?? "?") window \"\(title)\" (\(Int(wf.width))x\(Int(wf.height)))\n" + body)]
        if (args["screenshot"] as? Bool) ?? false {
            out.append(try await shot(app, wf))
        }
        return out
    }

    func shot(_ app: NSRunningApplication, _ wf: CGRect) async throws -> Content {
        let jpeg = try await Capture.window(pid: app.processIdentifier, frame: wf)
        return .image(base64: jpeg.base64EncodedString(), mime: "image/jpeg")
    }

    func screenshot(_ args: JSON) async throws -> [Content] {
        let app = try AX.findApp(try str(args, "app"))
        let window = try pickWindow(app, args["window"])
        guard let wf = AX.frame(window) else { throw ToolError(message: "The window has no frame") }
        return [.text("Screenshot of \(app.localizedName ?? "?") (\(Int(wf.width))x\(Int(wf.height))); x,y in it are what click and drag take."),
                try await shot(app, wf)]
    }

    func setValue(_ args: JSON) throws -> String {
        let app = try AX.findApp(try str(args, "app"))
        guard let r = args["ref"] as? Int, let el = state(app).element(ref: r) else {
            throw ToolError(message: "That ref is gone: the element is no longer shown. Call get_state and use a current id")
        }
        let value = args["value"] as? String ?? ""
        let role = AX.string(el, kAXRoleAttribute) ?? ""
        if role == "AXPopUpButton" || role == "AXMenuButton" {
            return try choose(el, value)
        }
        guard AX.settable(el, kAXValueAttribute) else {
            throw ToolError(message: "[\(r)] has no settable value; click it and type instead")
        }
        let err = AXUIElementSetAttributeValue(el, kAXValueAttribute as CFString, value as CFString)
        var now = settled(el) { $0 == value }
        if err == .success && now == value { return "Set [\(r)] to \"\(value)\"." }
        // Web text fields (WKWebView) can drop an accessibility write without an error: type it as keys instead.
        // Selecting the field's text through accessibility gives the page the caret (focus alone doesn't); typing
        // then replaces the selection.
        Background.focus(el)
        var range = CFRange(location: 0, length: (now as NSString).length)
        if let sel = AXValueCreate(.cfRange, &range),
           AXUIElementSetAttributeValue(el, kAXSelectedTextRangeAttribute as CFString, sel) == .success {
            usleep(60_000)
        } else {
            Background.key(0, flags: .maskCommand, pid: app.processIdentifier)  // cmd+a
        }
        if (now as NSString).length > 0 { Background.key(51, flags: [], pid: app.processIdentifier) }  // delete it
        Background.typeText(value, pid: app.processIdentifier)
        now = settled(el) { $0 == value }
        return now == value ? "Set [\(r)] to \"\(value)\" (typed as keys; the accessibility write didn't take)."
            : "Set [\(r)]; it now reads \"\(now)\". Check it before going on."
    }

    /// An element's value after a write, giving the app up to ~0.6 s to apply it.
    func settled(_ el: AXUIElement, until want: (String) -> Bool) -> String {
        var now = AX.string(el, kAXValueAttribute) ?? ""
        for _ in 0..<4 where !want(now) {
            usleep(150_000)
            now = AX.string(el, kAXValueAttribute) ?? ""
        }
        return now
    }

    /// Picks a pop-up's item by its title: opens its menu, presses the item.
    func choose(_ popup: AXUIElement, _ title: String) throws -> String {
        func reads() -> String { AX.string(popup, kAXValueAttribute) ?? AX.string(popup, kAXTitleAttribute) ?? "" }
        // Chromium lists a <select>'s options while it is closed: press the option itself, so no menu opens
        // (an open menu takes the keyboard from whatever app the user is in).
        let closed = AX.children(popup).filter { AX.string($0, kAXRoleAttribute) == "AXMenu" }.flatMap { AX.children($0) }
        if let item = closed.first(where: { AX.string($0, kAXTitleAttribute)?.caseInsensitiveCompare(title) == .orderedSame }),
           AXUIElementPerformAction(item, kAXPressAction as CFString) == .success {
            usleep(200_000)
            if reads().caseInsensitiveCompare(title) == .orderedSame { return "Chose \"\(title)\"; it now reads \"\(reads())\"." }
        }
        AXUIElementPerformAction(popup, kAXPressAction as CFString)
        usleep(250_000)
        let menus = AX.children(popup).filter { AX.string($0, kAXRoleAttribute) == "AXMenu" }
        let items = menus.flatMap { AX.children($0) }
        let titles = items.compactMap { AX.string($0, kAXTitleAttribute) }
        guard let item = items.first(where: { AX.string($0, kAXTitleAttribute)?.caseInsensitiveCompare(title) == .orderedSame }) else {
            if let m = menus.first { AXUIElementPerformAction(m, kAXCancelAction as CFString) }
            throw ToolError(message: "No item \"\(title)\". Items: \(titles.joined(separator: ", "))")
        }
        AXUIElementPerformAction(item, kAXPressAction as CFString)
        usleep(150_000)
        return "Chose \"\(title)\"; it now reads \"\(AX.string(popup, kAXValueAttribute) ?? AX.string(popup, kAXTitleAttribute) ?? "")\"."
    }

    /// The screen point an action targets: a ref's centre now, or x,y in the last snapshot's window.
    func target(_ app: NSRunningApplication, _ args: JSON, ref refKey: String = "ref", x xKey: String = "x",
                y yKey: String = "y") throws -> CGPoint {
        let st = state(app)
        if let r = args[refKey] as? Int {
            guard let el = st.element(ref: r) else { throw ToolError(message: "[\(r)] is no longer shown (nothing was done). Call get_state and use a current id") }
            guard let f = AX.frame(el) else { throw ToolError(message: "Ref [\(r)] has no position on screen") }
            return CGPoint(x: f.midX, y: f.midY)
        }
        guard let x = num(args[xKey]), let y = num(args[yKey]) else {
            throw ToolError(message: "Give \(refKey), or \(xKey) and \(yKey)")
        }
        guard let w = st.window ?? AX.windows(of: app.processIdentifier).first, let wf = AX.frame(w) else {
            throw ToolError(message: "Call get_state for \(app.localizedName ?? "the app") first")
        }
        return CGPoint(x: wf.minX + x, y: wf.minY + y)
    }

    /// The CGWindowID actions in an app go to: the element's window, else the window last read.
    func windowID(_ app: NSRunningApplication, _ el: AXUIElement? = nil) -> CGWindowID? {
        if let el, let w = Background.window(of: el), let id = Background.windowID(w) { return id }
        if let w = states[app.processIdentifier]?.window ?? AX.windows(of: app.processIdentifier).first {
            return Background.windowID(w)
        }
        return nil
    }

    /// Step 5: after an action, wait until the window stops changing (and any busy indicator clears, up to 4 s),
    /// then report what changed, so the agent sees the effect without another call.
    func afterAction(_ args: JSON, _ message: String) -> String {
        guard let name = args["app"] as? String, let app = try? AX.findApp(name) else { return message }
        let st = state(app)
        guard let window = (try? pickWindow(app, nil)) ?? st.window else { return message }
        func signature() -> Int {
            var h = Hasher()
            for n in AX.walk(window, maxNodes: 600) { h.combine(n.role); h.combine(n.value); h.combine(n.title) }
            return h.finalize()
        }
        func busy() -> Bool {
            AX.walk(window, maxNodes: 600).contains { $0.role == "AXBusyIndicator" || $0.role == "AXProgressIndicator" }
        }
        let start = Date()
        var last = signature(), stable = 0
        while Date().timeIntervalSince(start) < 4.0 {
            usleep(150_000)
            let now = signature()
            if now == last { stable += 1 } else { stable = 0; last = now }
            if stable >= 2 && (!busy() || Date().timeIntervalSince(start) > 4.0) { break }
        }
        let diff = st.snapshot(window: window, full: false, query: nil, frames: false)
        let lines = diff.split(separator: "\n", omittingEmptySubsequences: false)
        let shown = lines.count > 40 ? lines.prefix(40).joined(separator: "\n") + "\n… (\(lines.count - 40) more lines; get_state for all)" : diff
        return message + "\n" + shown
    }

    func click(_ args: JSON) throws -> String {
        let app = try AX.findApp(try str(args, "app"))
        let pid = app.processIdentifier
        let button = args["button"] as? String ?? "left"
        let count = args["count"] as? Int ?? 1
        var flags: CGEventFlags = []
        for m in (args["modifiers"] as? [String]) ?? [] { flags.formUnion(try Input.parseChord(m + "+a").1) }
        var el: AXUIElement?
        if let r = args["ref"] as? Int {
            guard let e = state(app).element(ref: r) else { throw ToolError(message: "[\(r)] is no longer shown (nothing was done). Call get_state and use a current id") }
            el = e
            if button == "left", count == 1, flags.isEmpty, let action = Background.press(e) {
                return "Pressed [\(r)] (\(action)) in the background."
            }
            Background.focus(e)
        }
        let p = try target(app, args)
        Background.click(at: p, pid: pid, window: windowID(app, el), button: button, count: count, flags: flags)
        return "Clicked at \(Int(p.x)),\(Int(p.y)) in the background; the app was not brought forward."
    }

    func typeText(_ args: JSON) throws -> String {
        let app = try AX.findApp(try str(args, "app"))
        let pid = app.processIdentifier
        let text = try str(args, "text")
        if let r = args["ref"] as? Int {
            guard let el = state(app).element(ref: r) else { throw ToolError(message: "[\(r)] is no longer shown (nothing was done). Call get_state and use a current id") }
            Background.focus(el)
            let before = AX.string(el, kAXValueAttribute) ?? ""
            if !text.contains("\n") && !text.contains("\t") && Background.insertText(el, text) {
                let now = settled(el) { $0.contains(text) }
                if now.contains(text) { return "Inserted \(text.count) characters into [\(r)] through accessibility." }
                // The write didn't reach the field (web text fields drop it silently): undo any partial effect, type keys.
                if now != before, AX.settable(el, kAXValueAttribute) {
                    AXUIElementSetAttributeValue(el, kAXValueAttribute as CFString, before as CFString)
                }
                usleep(60_000)
                Background.typeText(text, pid: pid)
                let after = settled(el) { $0.contains(text) }
                return after.contains(text) ? "Typed \(text.count) characters into [\(r)] as keys, in the background."
                    : "Typed \(text.count) characters at [\(r)] as keys, but it reads \"\(after)\". Check it before going on."
            }
        } else if args["x"] != nil {
            let p = try target(app, args)
            Background.click(at: p, pid: pid, window: windowID(app))
            usleep(80_000)
        }
        Background.typeText(text, pid: pid)
        return "Typed \(text.count) characters into the app in the background."
    }

    func pressKey(_ args: JSON) throws -> String {
        let app = try AX.findApp(try str(args, "app"))
        let (k, f) = try Input.parseChord(try str(args, "keys"))
        Background.key(k, flags: f, pid: app.processIdentifier)
        return "Pressed \(args["keys"] as! String) in the app, in the background."
    }

    func scroll(_ args: JSON) throws -> String {
        let app = try AX.findApp(try str(args, "app"))
        let p = try target(app, args)
        let amount = Int32(args["amount"] as? Int ?? 3)
        let dir = (args["direction"] as? String ?? "down").lowercased()
        let (dx, dy): (Int32, Int32) = switch dir {
        case "up": (0, amount)
        case "left": (amount, 0)
        case "right": (-amount, 0)
        default: (0, -amount)
        }
        Background.scroll(at: p, dx: dx, dy: dy, pid: app.processIdentifier, window: windowID(app))
        return "Scrolled \(dir) by \(amount) in the background."
    }

    func drag(_ args: JSON) throws -> String {
        let app = try AX.findApp(try str(args, "app"))
        let a = try target(app, args, ref: "from_ref", x: "from_x", y: "from_y")
        let b = try target(app, args, ref: "to_ref", x: "to_x", y: "to_y")
        Background.drag(from: a, to: b, pid: app.processIdentifier, window: windowID(app))
        return "Dragged from \(Int(a.x)),\(Int(a.y)) to \(Int(b.x)),\(Int(b.y)) in the background."
    }

    func bringToFront(_ args: JSON) throws -> String {
        let app = try AX.findApp(try str(args, "app"))
        Input.activate(app, window: states[app.processIdentifier]?.window)
        return "\(app.localizedName ?? "The app") is in front now."
    }

    func hover(_ args: JSON) throws -> String {
        let app = try AX.findApp(try str(args, "app"))
        guard app.isActive else {
            throw ToolError(message: "Hover moves the real pointer, so it only works on the app in front: call bring_to_front first (only when the task allows the foreground)")
        }
        let p = try target(app, args)
        Input.move(to: p)
        usleep(150_000)
        Input.move(to: CGPoint(x: p.x + 1, y: p.y))
        return "The pointer is over \(Int(p.x)),\(Int(p.y)). Wait about a second for a tooltip, then take a screenshot."
    }

    // MARK: dispatch

    func call(_ name: String, _ args: JSON) async -> (content: [Content], isError: Bool) {
        guard insideVM() else {
            return ([.text("Refused: this driver only runs inside a virtual machine, never on a real desktop.")], true)
        }
        guard AX.trusted() else {
            return ([.text("This driver lacks the Accessibility permission.")], true)
        }
        do {
            switch name {
            case "repl":
                guard let code = args["code"] as? String else { return ([.text("Missing \"code\"")], true) }
                return repl.run(code: code)
            case "list_apps": return ([.text(listApps())], false)
            case "get_state": return (try await getState(args), false)
            case "screenshot": return (try await screenshot(args), false)
            case "set_value": return ([.text(afterAction(args, try setValue(args)))], false)
            case "click": return ([.text(afterAction(args, try click(args)))], false)
            case "type_text": return ([.text(afterAction(args, try typeText(args)))], false)
            case "press_key": return ([.text(afterAction(args, try pressKey(args)))], false)
            case "scroll": return ([.text(afterAction(args, try scroll(args)))], false)
            case "drag": return ([.text(afterAction(args, try drag(args)))], false)
            case "bring_to_front": return ([.text(try bringToFront(args))], false)
            case "hover": return ([.text(try hover(args))], false)
            case "wait":
                let s = min(10.0, num(args["seconds"]) ?? 1.0)
                try await Task.sleep(nanoseconds: UInt64(s * 1e9))
                return ([.text("Waited \(s) s.")], false)
            default: return ([.text("Unknown tool \(name)")], true)
            }
        } catch let e as ToolError {
            return ([.text(e.message)], true)
        } catch {
            return ([.text("\(error)")], true)
        }
    }
}

/// Whether this process runs in a VM (the hypervisor flag macOS sets for guests). A safety guard: the driver must
/// never read or drive a real person's desktop.
func insideVM() -> Bool {
    var present: Int32 = 0
    var size = MemoryLayout<Int32>.size
    return sysctlbyname("kern.hv_vmm_present", &present, &size, nil, 0) == 0 && present == 1
}

func str(_ args: JSON, _ key: String) throws -> String {
    guard let s = args[key] as? String, !s.isEmpty else { throw ToolError(message: "Missing \"\(key)\"") }
    return s
}

func num(_ v: Any?) -> CGFloat? {
    if let d = v as? Double { return CGFloat(d) }
    if let i = v as? Int { return CGFloat(i) }
    if let n = v as? NSNumber { return CGFloat(n.doubleValue) }
    return nil
}

func schema(_ props: JSON, required: [String]) -> JSON {
    ["type": "object", "properties": props, "required": required]
}

let appProp: JSON = ["type": "string", "description": "The app's name, bundle id or pid"]
let refProp: JSON = ["type": "integer", "description": "An element's [N] ref from get_state"]
let xyNote = "x and y are in the app's window, from its top left, as in its screenshot"

let TOOLS: [JSON] = [
    ["name": "list_apps", "description": "List running apps and their windows.", "inputSchema": schema([:], required: [])],
    ["name": "get_state",
     "description": "Read one window of an app as a compact accessibility tree, each element with an [N] ref that stays the same while the element is shown. After the first look it returns only what changed (~ changed, + added, Removed); full=true gives the whole tree again, query=\"words\" only matching elements and their parents.",
     "inputSchema": schema(["app": appProp,
                            "window": ["description": "Window index from list_apps, or part of its title; default the focused window"],
                            "full": ["type": "boolean"], "query": ["type": "string"],
                            "frames": ["type": "boolean", "description": "Add each element's @x,y size in the window"],
                            "screenshot": ["type": "boolean", "description": "Also attach a screenshot (default false)"]],
                           required: ["app"])],
    ["name": "screenshot", "description": "A screenshot of an app's window, for what the tree can't show (canvases, images, tooltips). x,y in it are what click and drag take.",
     "inputSchema": schema(["app": appProp, "window": ["description": "Window index or part of its title"]], required: ["app"])],
    ["name": "set_value", "description": "Set a text field's value, or choose a pop-up button's item by its exact title, without typing.",
     "inputSchema": schema(["app": appProp, "ref": refProp, "value": ["type": "string"]], required: ["app", "ref", "value"])],
    ["name": "click", "description": "Click an element by ref (pressed through accessibility when it can be), or a point. \(xyNote). Works in the background: the app is not brought forward and the pointer does not move.",
     "inputSchema": schema(["app": appProp, "ref": refProp, "x": ["type": "number"], "y": ["type": "number"],
                            "button": ["type": "string", "enum": ["left", "right", "middle"]],
                            "count": ["type": "integer", "description": "1 to 3 clicks"],
                            "modifiers": ["type": "array", "items": ["type": "string"], "description": "cmd, shift, alt, ctrl"]],
                           required: ["app"])],
    ["name": "type_text", "description": "Type text into the app; with a ref or x,y, click there first. \\n presses Return, \\t Tab.",
     "inputSchema": schema(["app": appProp, "text": ["type": "string"], "ref": refProp, "x": ["type": "number"], "y": ["type": "number"]],
                           required: ["app", "text"])],
    ["name": "press_key", "description": "Press a key or chord, such as return, escape, tab, cmd+a, shift+tab.",
     "inputSchema": schema(["app": appProp, "keys": ["type": "string"]], required: ["app", "keys"])],
    ["name": "scroll", "description": "Scroll at an element or point.",
     "inputSchema": schema(["app": appProp, "ref": refProp, "x": ["type": "number"], "y": ["type": "number"],
                            "direction": ["type": "string", "enum": ["up", "down", "left", "right"]],
                            "amount": ["type": "integer", "description": "Lines, default 3"]], required: ["app"])],
    ["name": "drag", "description": "Drag with the left button from one element or point to another. \(xyNote).",
     "inputSchema": schema(["app": appProp, "from_ref": refProp, "from_x": ["type": "number"], "from_y": ["type": "number"],
                            "to_ref": refProp, "to_x": ["type": "number"], "to_y": ["type": "number"]], required: ["app"])],
    ["name": "bring_to_front", "description": "Bring an app to the front. Only when the task allows it: in the background it disturbs the user.",
     "inputSchema": schema(["app": appProp], required: ["app"])],
    ["name": "hover", "description": "Move the real pointer over an element or point, for hover-only controls and tooltips. Needs the app in front (bring_to_front), so use it only when the task allows the foreground.",
     "inputSchema": schema(["app": appProp, "ref": refProp, "x": ["type": "number"], "y": ["type": "number"]], required: ["app"])],
    ["name": "wait", "description": "Wait up to 10 seconds.",
     "inputSchema": schema(["seconds": ["type": "number"]], required: [])],
]

let INSTRUCTIONS = """
Operate macOS apps with these tools, in the background: they act on an app without bringing it forward, moving the \
pointer or using the clipboard. Call get_state on the app you need: a compact tree where each element has an [N] \
ref that stays valid while it is shown. Prefer set_value for text fields and pop-up buttons, and click by ref. After \
acting, call get_state again: it shows only what changed. Take a screenshot only when the tree can't tell you \
something. bring_to_front and hover take over the screen: use them only when the task allows the foreground.
"""

let REPL_TOOL: JSON = [
    "name": "repl",
    "description": "Run JavaScript whose only API is the `computer` global, which reads and controls native apps on this computer in the background. No file system, network or shell. Variables declared at the top level persist between calls. Start with `const app = await computer.getApp('<app name>')`; the guide is in the server instructions.",
    "inputSchema": schema(["code": ["type": "string", "description": "JavaScript to run; top-level await works"],
                           "title": ["type": "string", "description": "What the code does, in a few words"]],
                          required: ["code"]),
]

let REPL_INSTRUCTIONS = """
Operate macOS apps by writing JavaScript in the repl tool. The only API is `computer`:

  const app = await computer.getApp('BenchLab')   // prints the app's window as a tree; each element has an [N] id
  app.getAXState()                  // what changed since the last look (~ changed, + added, Removed)
  app.getAXState({ full: true })    // the whole tree again;  { query: 'words' } only matching elements
  app.click(N)  or  app.click([x, y])   // by id (pressed through accessibility when it can be) or a point
  app.setValue(N, 'text')           // set a text field, or choose a pop-up button's item by its exact title
  app.typeText('text', N)           // type into element N (or the focused element); \n is Return, \t is Tab
  app.pressKey('cmd+a')             // return, escape, tab, shift+tab, cmd+z, ...
  app.scroll(N, 'down', 3); app.drag(fromN or [x, y], toN or [x, y])
  app.getScreenshot()               // only when the tree can't tell you (images, canvases, tooltips)
  app.bringToFront(); app.hover(N)  // these take over the screen: only when the task allows the foreground
  computer.listApps(); sleep(ms); console.log(...)

Everything acts in the background: the app is not brought forward, the pointer doesn't move and the clipboard isn't \
touched. Every action waits for the app to settle and prints what changed, so you rarely \
need a separate look. Batch the steps you are sure of in one call. Ids stay valid while an element is shown. A failed action throws \
an Error with the reason: read it, look at the state, and don't repeat an action whose effect you haven't checked.
"""

if CommandLine.arguments.dropFirst().first == "check" {
    print("inside VM: \(insideVM()); accessibility trusted: \(AX.trusted())")
    if insideVM() { print(Driver().listApps()) }
    exit(0)
}

// A command-line process has no WindowServer connection until something opens one; ScreenCaptureKit and image
// encoding assert without it (CGS_REQUIRE_INIT). An NSApplication that never shows in the Dock opens it.
_ = NSApplication.shared
NSApp.setActivationPolicy(.prohibited)

let driver = Driver()
let server = MCPServer(name: "ourdriver", version: VERSION, instructions: REPL_MODE ? REPL_INSTRUCTIONS : INSTRUCTIONS,
                       tools: { REPL_MODE ? [REPL_TOOL] : TOOLS },
                       call: { name, args in await driver.call(name, args) })
await server.run()
