// Acting on apps (synthetic input through CGEvent) and seeing them (window screenshots through ScreenCaptureKit).
// Step 1 posts every event to the system's HID stream: the app comes to the front and the real pointer moves.
import AppKit
import CoreGraphics
import ScreenCaptureKit

enum Input {
    static let source = CGEventSource(stateID: .hidSystemState)

    /// Brings an app and one of its windows to the front, as a person clicking it would.
    static func activate(_ app: NSRunningApplication, window: AXUIElement?) {
        app.activate()
        if let w = window {
            AXUIElementPerformAction(w, kAXRaiseAction as CFString)
        }
        usleep(150_000)
    }

    static func mouse(_ type: CGEventType, _ p: CGPoint, _ button: CGMouseButton, clickState: Int64 = 1,
                      flags: CGEventFlags = []) {
        guard let e = CGEvent(mouseEventSource: source, mouseType: type, mouseCursorPosition: p, mouseButton: button)
        else { return }
        e.setIntegerValueField(.mouseEventClickState, value: clickState)
        e.flags = flags
        e.post(tap: .cghidEventTap)
    }

    static func click(at p: CGPoint, button: String = "left", count: Int = 1, flags: CGEventFlags = []) {
        let (down, up, b): (CGEventType, CGEventType, CGMouseButton) = switch button {
        case "right": (.rightMouseDown, .rightMouseUp, .right)
        case "middle": (.otherMouseDown, .otherMouseUp, .center)
        default: (.leftMouseDown, .leftMouseUp, .left)
        }
        mouse(.mouseMoved, p, b)
        usleep(30_000)
        for i in 1...max(1, count) {
            mouse(down, p, b, clickState: Int64(i), flags: flags)
            usleep(20_000)
            mouse(up, p, b, clickState: Int64(i), flags: flags)
            usleep(40_000)
        }
    }

    static func move(to p: CGPoint) { mouse(.mouseMoved, p, .left) }

    static func drag(from a: CGPoint, to b: CGPoint, steps: Int = 20) {
        mouse(.mouseMoved, a, .left)
        usleep(50_000)
        mouse(.leftMouseDown, a, .left)
        usleep(80_000)
        for i in 1...steps {
            let t = CGFloat(i) / CGFloat(steps)
            mouse(.leftMouseDragged, CGPoint(x: a.x + (b.x - a.x) * t, y: a.y + (b.y - a.y) * t), .left)
            usleep(15_000)
        }
        usleep(80_000)
        mouse(.leftMouseUp, b, .left)
    }

    static func scroll(at p: CGPoint, dx: Int32, dy: Int32) {
        move(to: p)
        usleep(30_000)
        guard let e = CGEvent(scrollWheelEvent2Source: source, units: .line, wheelCount: 2, wheel1: dy, wheel2: dx,
                              wheel3: 0) else { return }
        e.post(tap: .cghidEventTap)
    }

    /// Types text as Unicode key events, a few characters per event; newlines press Return, tabs press Tab.
    static func typeText(_ text: String) {
        var buffer: [UniChar] = []
        func flush() {
            var i = 0
            while i < buffer.count {
                let chunk = Array(buffer[i..<min(i + 16, buffer.count)])
                for down in [true, false] {
                    guard let e = CGEvent(keyboardEventSource: source, virtualKey: 0, keyDown: down) else { continue }
                    e.flags = []
                    e.keyboardSetUnicodeString(stringLength: chunk.count, unicodeString: chunk)
                    e.post(tap: .cghidEventTap)
                }
                usleep(8_000)
                i += 16
            }
            buffer.removeAll()
        }
        for ch in text {
            if ch == "\n" { flush(); tap(keyCode: 36, flags: []) }
            else if ch == "\t" { flush(); tap(keyCode: 48, flags: []) }
            else { buffer.append(contentsOf: Array(String(ch).utf16)) }
        }
        flush()
    }

    static func tap(keyCode: CGKeyCode, flags: CGEventFlags) {
        for down in [true, false] {
            guard let e = CGEvent(keyboardEventSource: source, virtualKey: keyCode, keyDown: down) else { continue }
            e.flags = flags
            e.post(tap: .cghidEventTap)
            usleep(15_000)
        }
    }

    static let keyCodes: [String: CGKeyCode] = [
        "a": 0, "s": 1, "d": 2, "f": 3, "h": 4, "g": 5, "z": 6, "x": 7, "c": 8, "v": 9, "b": 11, "q": 12, "w": 13,
        "e": 14, "r": 15, "y": 16, "t": 17, "1": 18, "2": 19, "3": 20, "4": 21, "6": 22, "5": 23, "=": 24, "9": 25,
        "7": 26, "-": 27, "8": 28, "0": 29, "]": 30, "o": 31, "u": 32, "[": 33, "i": 34, "p": 35, "l": 37, "j": 38,
        "'": 39, "k": 40, ";": 41, "\\": 42, ",": 43, "/": 44, "n": 45, "m": 46, ".": 47, "`": 50,
        "return": 36, "enter": 36, "tab": 48, "space": 49, "delete": 51, "backspace": 51, "escape": 53, "esc": 53,
        "forwarddelete": 117, "home": 115, "end": 119, "pageup": 116, "pagedown": 121,
        "left": 123, "right": 124, "down": 125, "up": 126,
        "f1": 122, "f2": 120, "f3": 99, "f4": 118, "f5": 96, "f6": 97, "f7": 98, "f8": 100, "f9": 101, "f10": 109,
        "f11": 103, "f12": 111,
    ]

    /// A chord such as "cmd+shift+z" or "return": modifiers, then one key.
    static func parseChord(_ chord: String) throws -> (CGKeyCode, CGEventFlags) {
        var flags: CGEventFlags = []
        var key: CGKeyCode?
        for raw in chord.lowercased().split(separator: "+").map({ $0.trimmingCharacters(in: .whitespaces) }) {
            switch raw {
            case "cmd", "command", "super", "meta": flags.insert(.maskCommand)
            case "shift": flags.insert(.maskShift)
            case "alt", "option", "opt": flags.insert(.maskAlternate)
            case "ctrl", "control": flags.insert(.maskControl)
            case "fn": flags.insert(.maskSecondaryFn)
            default:
                guard let k = keyCodes[raw] else { throw ToolError(message: "Unknown key \"\(raw)\" in \"\(chord)\"") }
                key = k
            }
        }
        guard let k = key else { throw ToolError(message: "No key in \"\(chord)\"") }
        return (k, flags)
    }
}

enum Capture {
    /// A window's image as JPEG, at one pixel per point, found by its owner's pid and its frame.
    static func window(pid: pid_t, frame: CGRect) async throws -> Data {
        let content = try await SCShareableContent.excludingDesktopWindows(false, onScreenWindowsOnly: false)
        let mine = content.windows.filter { $0.owningApplication?.processID == pid && $0.windowLayer == 0 }
        guard let w = mine.min(by: { distance($0.frame, frame) < distance($1.frame, frame) }) else {
            throw ToolError(message: "No capturable window for pid \(pid)")
        }
        let filter = SCContentFilter(desktopIndependentWindow: w)
        let cfg = SCStreamConfiguration()
        cfg.width = max(1, Int(w.frame.width))
        cfg.height = max(1, Int(w.frame.height))
        cfg.showsCursor = false
        let image = try await SCScreenshotManager.captureImage(contentFilter: filter, configuration: cfg)
        let rep = NSBitmapImageRep(cgImage: image)
        guard let jpeg = rep.representation(using: .jpeg, properties: [.compressionFactor: 0.6]) else {
            throw ToolError(message: "Could not encode the screenshot")
        }
        return jpeg
    }

    static func distance(_ a: CGRect, _ b: CGRect) -> CGFloat {
        abs(a.minX - b.minX) + abs(a.minY - b.minY) + abs(a.width - b.width) + abs(a.height - b.height)
    }
}
