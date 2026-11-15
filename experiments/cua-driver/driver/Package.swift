// swift-tools-version:5.10
import PackageDescription

let package = Package(
    name: "OurDriver",
    platforms: [.macOS(.v14)],
    targets: [
        .executableTarget(
            name: "ourdriver",
            path: "Sources/OurDriver",
            linkerSettings: [
                .linkedFramework("ApplicationServices"),
                .linkedFramework("AppKit"),
                .linkedFramework("ScreenCaptureKit"),
            ]
        ),
    ]
)
