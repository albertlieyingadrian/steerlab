// A minimal MCP server over stdio: newline-delimited JSON-RPC 2.0, the methods a tool server needs.
import Foundation

typealias JSON = [String: Any]

struct ToolError: Error { let message: String }

/// One content block of a tool result: text, or an image as base64.
enum Content {
    case text(String)
    case image(base64: String, mime: String)

    var json: JSON {
        switch self {
        case .text(let t): return ["type": "text", "text": t]
        case .image(let b, let m): return ["type": "image", "data": b, "mimeType": m]
        }
    }
}

final class MCPServer {
    let name: String
    let version: String
    let instructions: String
    let tools: () -> [JSON]
    let call: (String, JSON) async -> (content: [Content], isError: Bool)

    init(name: String, version: String, instructions: String, tools: @escaping () -> [JSON],
         call: @escaping (String, JSON) async -> (content: [Content], isError: Bool)) {
        self.name = name
        self.version = version
        self.instructions = instructions
        self.tools = tools
        self.call = call
    }

    func run() async {
        do {
            for try await line in FileHandle.standardInput.bytes.lines {
                let trimmed = line.trimmingCharacters(in: .whitespaces)
                if trimmed.isEmpty { continue }
                guard let data = trimmed.data(using: .utf8),
                      let msg = try? JSONSerialization.jsonObject(with: data) as? JSON else {
                    send(["jsonrpc": "2.0", "id": NSNull(), "error": ["code": -32700, "message": "parse error"]])
                    continue
                }
                await handle(msg)
            }
        } catch {
            log("stdin ended: \(error)")
        }
    }

    private func handle(_ msg: JSON) async {
        guard let method = msg["method"] as? String else { return }  // a response to us; we send no requests
        let id = msg["id"]
        let params = msg["params"] as? JSON ?? [:]
        switch method {
        case "initialize":
            let requested = params["protocolVersion"] as? String ?? "2025-06-18"
            reply(id, [
                "protocolVersion": requested,
                "capabilities": ["tools": ["listChanged": false]],
                "serverInfo": ["name": name, "version": version],
                "instructions": instructions,
            ])
        case "ping":
            reply(id, [:])
        case "tools/list":
            reply(id, ["tools": tools()])
        case "tools/call":
            let tool = params["name"] as? String ?? ""
            let args = params["arguments"] as? JSON ?? [:]
            let result = await call(tool, args)
            reply(id, ["content": result.content.map { $0.json }, "isError": result.isError])
        default:
            if let id {
                send(["jsonrpc": "2.0", "id": id, "error": ["code": -32601, "message": "method not found: \(method)"]])
            }
        }
    }

    private func reply(_ id: Any?, _ result: JSON) {
        guard let id else { return }
        send(["jsonrpc": "2.0", "id": id, "result": result])
    }

    private func send(_ obj: JSON) {
        guard let data = try? JSONSerialization.data(withJSONObject: obj, options: [.withoutEscapingSlashes]) else { return }
        FileHandle.standardOutput.write(data)
        FileHandle.standardOutput.write("\n".data(using: .utf8)!)
    }
}

func log(_ s: String) {
    FileHandle.standardError.write("ourdriver: \(s)\n".data(using: .utf8)!)
}
