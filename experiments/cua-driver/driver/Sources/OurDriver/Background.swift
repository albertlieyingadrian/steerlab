// Step 3: acting on an app without taking the user's pointer, keyboard or front window.
// Order of preference: an accessibility action or value (no events at all), then events posted to the app's process
// (CGEventPostToPid, never the system's HID stream, so the real pointer doesn't move and nothing is activated).
import AppKit
import ApplicationServices

/// The CGWindowID of an AX window (private, stable since 10.x; cua-driver and others use it).
@_silgen_name("_AXUIElementGetWindow")
func _AXUIElementGetWindow(_ element: AXUIElement, _ id: UnsafeMutablePointer<CGWindowID>) -> AXError

enum Background {
    /// A private event source, so events carry none of the user's held keys or buttons.
    static let source = CGEventSource(stateID: .privateState)

    static func windowID(_ window: AXUIElement) -> CGWindowID? {
        var id: CGWindowID = 0
        return _AXUIElementGetWindow(window, &id) == .success && id != 0 ? id : nil
    }

    /// The window an element is in.
    static func window(of el: AXUIElement) -> AXUIElement? {
        if let w = AX.attr(el, kAXWindowAttribute) { return (w as! AXUIElement) }
        if AX.string(el, kAXRoleAttribute) == "AXWindow" { return el }
        return nil
    }

    // MARK: accessibility first

    /// Presses an element through accessibility if it offers a press-like action. Returns the action used.
    static func press(_ el: AXUIElement) -> String? {
        let actions = AX.actions(el)
        for a in ["AXPress", "AXConfirm", "AXPick", "AXOpen"] where actions.contains(a) {
            if AXUIElementPerformAction(el, a as CFString) == .success { return a }
        }
        return nil
    }

    /// Gives an element the keyboard focus inside its app, without activating the app.
    static func focus(_ el: AXUIElement) {
        AXUIElementSetAttributeValue(el, kAXFocusedAttribute as CFString, kCFBooleanTrue)
    }

    /// Inserts text at an element's caret through accessibility, or appends it to its value. True if it worked.
    static func insertText(_ el: AXUIElement, _ text: String) -> Bool {
        if AX.settable(el, kAXSelectedTextAttribute),
           AXUIElementSetAttributeValue(el, kAXSelectedTextAttribute as CFString, text as CFString) == .success {
            return true
        }
        if AX.settable(el, kAXValueAttribute) {
            let now = AX.string(el, kAXValueAttribute) ?? ""
            return AXUIElementSetAttributeValue(el, kAXValueAttribute as CFString, (now + text) as CFString) == .success
        }
        return false
    }

    // MARK: events to the process

    static func post(_ e: CGEvent, pid: pid_t, window: CGWindowID?) {
        if let w = window {
            e.setIntegerValueField(.mouseEventWindowUnderMousePointer, value: Int64(w))
            e.setIntegerValueField(.mouseEventWindowUnderMousePointerThatCanHandleThisEvent, value: Int64(w))
        }
        e.postToPid(pid)
    }

    static func mouse(_ type: CGEventType, _ p: CGPoint, _ b: CGMouseButton, pid: pid_t, window: CGWindowID?,
                      clickState: Int64 = 1, flags: CGEventFlags = []) {
        guard let e = CGEvent(mouseEventSource: source, mouseType: type, mouseCursorPosition: p, mouseButton: b)
        else { return }
        e.setIntegerValueField(.mouseEventClickState, value: clickState)
        e.flags = flags
        post(e, pid: pid, window: window)
    }

    /// A click at a screen point, delivered to the app's process. A leading mouse-moved primes AppKit's hit test.
    static func click(at p: CGPoint, pid: pid_t, window: CGWindowID?, button: String = "left", count: Int = 1,
                      flags: CGEventFlags = []) {
        let (down, up, b): (CGEventType, CGEventType, CGMouseButton) = switch button {
        case "right": (.rightMouseDown, .rightMouseUp, .right)
        case "middle": (.otherMouseDown, .otherMouseUp, .center)
        default: (.leftMouseDown, .leftMouseUp, .left)
        }
        mouse(.mouseMoved, p, b, pid: pid, window: window)
        usleep(30_000)
        for i in 1...max(1, count) {
            mouse(down, p, b, pid: pid, window: window, clickState: Int64(i), flags: flags)
            usleep(28_000)
            mouse(up, p, b, pid: pid, window: window, clickState: Int64(i), flags: flags)
            usleep(40_000)
        }
    }

    /// A drag inside one window, delivered to the app's process.
    static func drag(from a: CGPoint, to b: CGPoint, pid: pid_t, window: CGWindowID?, steps: Int = 24) {
        mouse(.mouseMoved, a, .left, pid: pid, window: window)
        usleep(40_000)
        mouse(.leftMouseDown, a, .left, pid: pid, window: window)
        usleep(120_000)
        for i in 1...steps {
            let t = CGFloat(i) / CGFloat(steps)
            let p = CGPoint(x: a.x + (b.x - a.x) * t, y: a.y + (b.y - a.y) * t)
            mouse(.leftMouseDragged, p, .left, pid: pid, window: window)
            usleep(16_000)
        }
        usleep(150_000)
        mouse(.leftMouseUp, b, .left, pid: pid, window: window)
    }

    static func scroll(at p: CGPoint, dx: Int32, dy: Int32, pid: pid_t, window: CGWindowID?) {
        guard let e = CGEvent(scrollWheelEvent2Source: source, units: .line, wheelCount: 2, wheel1: dy, wheel2: dx,
                              wheel3: 0) else { return }
        e.location = p
        post(e, pid: pid, window: window)
    }

    static func key(_ code: CGKeyCode, flags: CGEventFlags, pid: pid_t) {
        for down in [true, false] {
            guard let e = CGEvent(keyboardEventSource: source, virtualKey: code, keyDown: down) else { continue }
            e.flags = flags
            e.postToPid(pid)
            usleep(12_000)
        }
    }

    /// Types text as Unicode key events to the app's process; newline presses Return, tab presses Tab.
    static func typeText(_ text: String, pid: pid_t) {
        var buffer: [UniChar] = []
        func flush() {
            var i = 0
            while i < buffer.count {
                let chunk = Array(buffer[i..<min(i + 16, buffer.count)])
                for down in [true, false] {
                    guard let e = CGEvent(keyboardEventSource: source, virtualKey: 0, keyDown: down) else { continue }
                    e.keyboardSetUnicodeString(stringLength: chunk.count, unicodeString: chunk)
                    e.postToPid(pid)
                }
                usleep(8_000)
                i += 16
            }
            buffer.removeAll()
        }
        for ch in text {
            if ch == "\n" { flush(); key(36, flags: [], pid: pid) }
            else if ch == "\t" { flush(); key(48, flags: [], pid: pid) }
            else { buffer.append(contentsOf: Array(String(ch).utf16)) }
        }
        flush()
    }
}
