// Step 2: stable refs and compact snapshots.
// An element keeps its [N] for as long as the app shows it, across snapshots; a snapshot after the first is a diff;
// the tree drops the containers a person never sees as things, and roles read as words.
import AppKit
import ApplicationServices

/// An AX element as a dictionary key: the same on-screen element compares equal across queries.
final class ElementKey: Hashable {
    let element: AXUIElement
    init(_ e: AXUIElement) { element = e }
    static func == (a: ElementKey, b: ElementKey) -> Bool { CFEqual(a.element, b.element) }
    func hash(into h: inout Hasher) { h.combine(CFHash(element)) }
}

/// What the model was last shown of one app.
final class AppState {
    let pid: pid_t
    private var ids: [ElementKey: Int] = [:]
    private var elements: [Int: AXUIElement] = [:]
    private var next = 1
    /// The lines of the last snapshot shown, by id, in order; and the window it was of.
    private(set) var shown: [(id: Int, line: String)] = []
    private(set) var window: AXUIElement?

    init(pid: pid_t) { self.pid = pid }

    func id(for el: AXUIElement) -> Int {
        let key = ElementKey(el)
        if let i = ids[key] { return i }
        let i = next
        next += 1
        ids[key] = i
        elements[i] = el
        return i
    }

    /// What each id was (role and label), to find it again if the app rebuilt it.
    private var keys: [Int: String] = [:]

    /// The element behind an id. If the app replaced that element (the same control rebuilt), the id moves to the
    /// element now shown with the same role and label.
    func element(ref: Int) -> AXUIElement? {
        guard let el = elements[ref] else { return nil }
        var role: AnyObject?
        if AXUIElementCopyAttributeValue(el, kAXRoleAttribute as CFString, &role) != .invalidUIElement { return el }
        guard let key = keys[ref], let w = window else { return nil }
        for n in AX.walk(w, maxNodes: 1500) where Compact.key(n) == key {
            elements[ref] = n.element
            ids[ElementKey(n.element)] = ref
            return n.element
        }
        return nil
    }

    func remember(_ id: Int, _ n: Node) { keys[id] = Compact.key(n) }

    /// The window's tree as compact lines: [N] role "label" value, with containers that carry nothing left out
    /// (their children move up a level).
    func lines(of window: AXUIElement, frames: Bool) -> [(id: Int, line: String)] {
        let origin = AX.frame(window)?.origin ?? .zero
        var out: [(Int, String)] = []
        func visit(_ el: AXUIElement, _ depth: Int, _ budget: inout Int) {
            if budget <= 0 { return }
            let n = AX.node(el, depth: depth)
            let kids = AX.children(el)
            let keep = Compact.keep(n, childCount: kids.count)
            var d = depth
            if keep {
                budget -= 1
                let i = id(for: el)
                remember(i, n)
                out.append((i, String(repeating: "  ", count: depth) + "[\(i)] " + Compact.describe(n, origin: origin, frames: frames)))
                d = depth + 1
            }
            if Compact.skipChildren(n) { return }
            for c in kids { visit(c, d, &budget) }
        }
        var budget = 1500
        visit(window, 0, &budget)
        return out
    }

    /// The full snapshot, or what changed since the last one shown of the same window.
    func snapshot(window: AXUIElement, full: Bool, query: String?, frames: Bool) -> String {
        let now = lines(of: window, frames: frames)
        let sameWindow = self.window.map { CFEqual($0, window) } ?? false
        defer { shown = now; self.window = window }
        if let q = query?.lowercased(), !q.isEmpty {
            return Compact.filter(now, query: q)
        }
        if full || !sameWindow || shown.isEmpty {
            return now.map { $0.line }.joined(separator: "\n")
        }
        let before = Dictionary(shown.map { ($0.id, $0.line) }, uniquingKeysWith: { a, _ in a })
        let after = Set(now.map { $0.id })
        var changes: [String] = []
        for (i, line) in now {
            if let old = before[i] {
                if old != line { changes.append("~ " + line.trimmingCharacters(in: .whitespaces)) }
            } else {
                changes.append("+ " + line)
            }
        }
        let removed = shown.map { $0.id }.filter { !after.contains($0) }
        if changes.isEmpty && removed.isEmpty { return "No change since the last look." }
        if changes.count + removed.count > max(12, now.count * 6 / 10) {
            return now.map { $0.line }.joined(separator: "\n")   // mostly new: the whole tree reads better
        }
        var text = "Changes since the last look (~ changed, + added):\n" + changes.joined(separator: "\n")
        if !removed.isEmpty {
            text += "\nRemoved: " + removed.map { "[\($0)]" }.joined(separator: " ")
        }
        return text
    }
}

enum Compact {
    static let names: [String: String] = [
        "AXStaticText": "text", "AXTextField": "text field", "AXTextArea": "text area", "AXButton": "button",
        "AXCheckBox": "checkbox", "AXRadioButton": "radio button", "AXPopUpButton": "pop-up button",
        "AXMenuButton": "menu button", "AXComboBox": "combo box", "AXSlider": "slider", "AXTable": "table",
        "AXOutline": "outline", "AXRow": "row", "AXColumn": "column", "AXCell": "cell", "AXList": "list",
        "AXScrollArea": "scroll area", "AXWindow": "window", "AXSheet": "sheet", "AXDialog": "dialog",
        "AXImage": "image", "AXLink": "link", "AXTabGroup": "tab group", "AXToolbar": "toolbar",
        "AXMenu": "menu", "AXMenuItem": "menu item", "AXMenuBar": "menu bar", "AXMenuBarItem": "menu",
        "AXHeading": "heading", "AXWebArea": "web page", "AXGroup": "group", "AXSplitGroup": "split group",
        "AXIncrementor": "stepper", "AXDisclosureTriangle": "disclosure", "AXProgressIndicator": "progress",
        "AXBusyIndicator": "busy", "AXValueIndicator": "value indicator", "AXSearchField": "search field",
        "AXLayoutArea": "layout area", "AXSplitter": "splitter", "AXScrollBar": "scroll bar",
    ]

    /// An element's identity for finding it again: role and label, not value.
    static func key(_ n: Node) -> String { n.role + "|" + (n.title ?? n.desc ?? n.identifier ?? "") }

    static func role(_ n: Node) -> String {
        if n.subrole == "AXSearchField" { return "search field" }
        if n.subrole == "AXSecureTextField" { return "password field" }
        if n.subrole == "AXCloseButton" { return "close button" }
        if n.subrole == "AXMinimizeButton" { return "minimize button" }
        if n.subrole == "AXFullScreenButton" || n.subrole == "AXZoomButton" { return "zoom button" }
        if n.subrole == "AXStandardWindow" { return "window" }
        if n.subrole == "AXDialog" || n.subrole == "AXSystemDialog" { return "dialog" }
        return names[n.role] ?? n.role.replacingOccurrences(of: "AX", with: "").lowercased()
    }

    static let containers: Set<String> = ["AXGroup", "AXSplitGroup", "AXLayoutArea", "AXUnknown", "AXScrollArea",
                                          "AXLayoutItem", "AXGenericElement"]

    /// Whether a node is worth a line: anything with a label, a value or an action, and every control.
    static func keep(_ n: Node, childCount: Int) -> Bool {
        if n.role == "AXScrollBar" || n.role == "AXSplitter" || n.role == "AXValueIndicator" { return false }
        if n.subrole == "AXMinimizeButton" || n.subrole == "AXFullScreenButton" || n.subrole == "AXZoomButton" {
            return false
        }
        if let f = n.frame, f.width < 1 || f.height < 1, n.role != "AXWindow" { return false }
        if containers.contains(n.role) {
            return n.title != nil || n.desc != nil || n.identifier.map { !$0.hasPrefix("_NS") } == true
                || n.actions.contains("AXPress")
        }
        return true
    }

    /// Children a node's own line already says: a text's runs, a pop-up's closed menu, a scroll bar's parts.
    static func skipChildren(_ n: Node) -> Bool {
        n.role == "AXStaticText" || n.role == "AXPopUpButton" || n.role == "AXScrollBar"
            || n.role == "AXMenuButton" || n.role == "AXImage"
    }

    static func describe(_ n: Node, origin: CGPoint, frames: Bool) -> String {
        var s = role(n)
        if !n.enabled { s += " (disabled)" }
        if n.settable && (n.role == "AXTextField" || n.role == "AXTextArea" || n.role == "AXComboBox") {
            s += " (settable)"
        }
        let label = n.title ?? n.desc
        if let l = label { s += " \(l)" }
        if let v = n.value, v != label {
            if n.role == "AXCheckBox" || n.role == "AXRadioButton" { s += v == "1" ? ", checked" : ", unchecked" }
            else { s += ", value: \(v)" }
        }
        if n.desc != nil, n.title != nil, n.desc != n.title { s += " (\(n.desc!))" }
        if n.focused { s += " (focused)" }
        if frames, let f = n.frame {
            s += " @\(Int(f.minX - origin.x)),\(Int(f.minY - origin.y)) \(Int(f.width))x\(Int(f.height))"
        }
        return s
    }

    /// The lines whose text matches a query, with their ancestors for context.
    static func filter(_ lines: [(id: Int, line: String)], query: String) -> String {
        var keep = Set<Int>()
        var stack: [(indent: Int, idx: Int)] = []
        for (idx, l) in lines.enumerated() {
            let indent = l.line.prefix(while: { $0 == " " }).count
            while let last = stack.last, last.indent >= indent { stack.removeLast() }
            if l.line.lowercased().contains(query) {
                keep.insert(idx)
                stack.forEach { keep.insert($0.idx) }
            }
            stack.append((indent, idx))
        }
        if keep.isEmpty { return "Nothing matches \"\(query)\" (\(lines.count) elements looked at)." }
        return lines.indices.filter { keep.contains($0) }.map { lines[$0].line }.joined(separator: "\n")
    }
}
