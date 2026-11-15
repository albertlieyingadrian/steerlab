// Reading apps through the macOS accessibility (AX) API: running apps, their windows, and each window's element tree.
import AppKit
import ApplicationServices

/// One element in a snapshot, with what the model sees of it.
struct Node {
    let element: AXUIElement
    let depth: Int
    let role: String
    let subrole: String?
    let title: String?
    let value: String?
    let desc: String?
    let identifier: String?
    let frame: CGRect?          // screen points, top-left origin
    let enabled: Bool
    let focused: Bool
    let settable: Bool
    let actions: [String]
}

enum AX {
    static func trusted() -> Bool { AXIsProcessTrusted() }

    static func attr(_ el: AXUIElement, _ name: String) -> AnyObject? {
        var v: AnyObject?
        return AXUIElementCopyAttributeValue(el, name as CFString, &v) == .success ? v : nil
    }

    static func string(_ el: AXUIElement, _ name: String) -> String? {
        guard let v = attr(el, name) else { return nil }
        if let s = v as? String { return s.isEmpty ? nil : s }
        if let n = v as? NSNumber { return n.stringValue }
        if let u = v as? URL { return u.absoluteString }
        return nil
    }

    static func bool(_ el: AXUIElement, _ name: String) -> Bool? {
        (attr(el, name) as? NSNumber)?.boolValue
    }

    static func children(_ el: AXUIElement, _ name: String = kAXChildrenAttribute) -> [AXUIElement] {
        (attr(el, name) as? [AXUIElement]) ?? []
    }

    static func frame(_ el: AXUIElement) -> CGRect? {
        guard let p = attr(el, kAXPositionAttribute), let s = attr(el, kAXSizeAttribute) else { return nil }
        var point = CGPoint.zero, size = CGSize.zero
        guard AXValueGetValue(p as! AXValue, .cgPoint, &point), AXValueGetValue(s as! AXValue, .cgSize, &size) else {
            return nil
        }
        return CGRect(origin: point, size: size)
    }

    static func settable(_ el: AXUIElement, _ name: String) -> Bool {
        var b: DarwinBoolean = false
        return AXUIElementIsAttributeSettable(el, name as CFString, &b) == .success && b.boolValue
    }

    static func actions(_ el: AXUIElement) -> [String] {
        var names: CFArray?
        guard AXUIElementCopyActionNames(el, &names) == .success, let a = names as? [String] else { return [] }
        return a
    }

    /// Apps already asked to expose their full tree.
    nonisolated(unsafe) static var exposed: Set<pid_t> = []

    static func app(pid: pid_t) -> AXUIElement {
        let a = AXUIElementCreateApplication(pid)
        AXUIElementSetMessagingTimeout(a, 2.0)
        if !exposed.contains(pid) {
            // Chromium and Electron build their accessibility tree only for assistive tech that asks for it.
            AXUIElementSetAttributeValue(a, "AXManualAccessibility" as CFString, kCFBooleanTrue)
            AXUIElementSetAttributeValue(a, "AXEnhancedUserInterface" as CFString, kCFBooleanTrue)
            exposed.insert(pid)
        }
        return a
    }

    /// The running apps a person could see: regular activation policy, by name.
    static func runningApps() -> [NSRunningApplication] {
        NSWorkspace.shared.runningApplications.filter { $0.activationPolicy == .regular }
    }

    /// The app a name, bundle id or pid names, case-insensitively.
    static func findApp(_ query: String) throws -> NSRunningApplication {
        let apps = runningApps()
        let q = query.lowercased()
        if let pid = Int32(query), let a = apps.first(where: { $0.processIdentifier == pid }) { return a }
        if let a = apps.first(where: { $0.localizedName?.lowercased() == q || $0.bundleIdentifier?.lowercased() == q }) {
            return a
        }
        if let a = apps.first(where: { ($0.localizedName?.lowercased() ?? "").contains(q) }) { return a }
        let names = apps.compactMap { $0.localizedName }.joined(separator: ", ")
        throw ToolError(message: "No running app matches \"\(query)\". Running apps: \(names)")
    }

    static func windows(of pid: pid_t) -> [AXUIElement] {
        children(app(pid: pid), kAXWindowsAttribute)
    }

    static func node(_ el: AXUIElement, depth: Int) -> Node {
        let role = string(el, kAXRoleAttribute) ?? "AXUnknown"
        var value = string(el, kAXValueAttribute)
        if let v = value, v.count > 300 { value = String(v.prefix(300)) + "…(\(v.count) chars)" }
        return Node(
            element: el, depth: depth, role: role,
            subrole: string(el, kAXSubroleAttribute),
            title: string(el, kAXTitleAttribute),
            value: value,
            desc: string(el, kAXDescriptionAttribute) ?? string(el, kAXHelpAttribute),
            identifier: string(el, "AXIdentifier"),
            frame: frame(el),
            enabled: bool(el, kAXEnabledAttribute) ?? true,
            focused: bool(el, kAXFocusedAttribute) ?? false,
            settable: settable(el, kAXValueAttribute),
            actions: actions(el))
    }

    /// Depth-first walk of an element's subtree, bounded in nodes and depth.
    static func walk(_ root: AXUIElement, maxNodes: Int = 2000, maxDepth: Int = 40) -> [Node] {
        var out: [Node] = []
        func visit(_ el: AXUIElement, _ depth: Int) {
            if out.count >= maxNodes || depth > maxDepth { return }
            out.append(node(el, depth: depth))
            for c in children(el) { visit(c, depth + 1) }
        }
        visit(root, 0)
        return out
    }
}
