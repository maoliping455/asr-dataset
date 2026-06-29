import Foundation
import Speech

struct Options {
    var audioPath: String = ""
    var localeID: String = "en-US"
    var onDevice: Bool = true
    var timeoutSec: Double = 180.0
    var outPath: String?
}

func jsonEscape(_ value: String) -> String {
    let data = try! JSONSerialization.data(withJSONObject: [value], options: [])
    let wrapped = String(data: data, encoding: .utf8)!
    return String(wrapped.dropFirst().dropLast())
}

func printJSON(_ values: [String: Any]) {
    let data = try! JSONSerialization.data(withJSONObject: values, options: [.prettyPrinted, .sortedKeys])
    print(String(data: data, encoding: .utf8)!)
}

func emitJSON(_ values: [String: Any], outPath: String?) {
    let data = try! JSONSerialization.data(withJSONObject: values, options: [.prettyPrinted, .sortedKeys])
    if let outPath = outPath {
        try! data.write(to: URL(fileURLWithPath: outPath))
    } else {
        print(String(data: data, encoding: .utf8)!)
    }
}

func parseArgs() -> Options {
    var options = Options()
    var index = 1
    let args = CommandLine.arguments
    while index < args.count {
        let arg = args[index]
        switch arg {
        case "--audio":
            index += 1
            options.audioPath = args[index]
        case "--locale":
            index += 1
            options.localeID = args[index]
        case "--on-device":
            options.onDevice = true
        case "--allow-server":
            options.onDevice = false
        case "--timeout-sec":
            index += 1
            options.timeoutSec = Double(args[index]) ?? options.timeoutSec
        case "--out":
            index += 1
            options.outPath = args[index]
        default:
            fputs("unknown argument: \(arg)\n", stderr)
            exit(2)
        }
        index += 1
    }
    if options.audioPath.isEmpty {
        fputs("usage: apple_speech_transcribe --audio file.wav --locale en-US [--on-device|--allow-server]\n", stderr)
        exit(2)
    }
    return options
}

func requestSpeechAuthorization(timeoutSec: Double) -> SFSpeechRecognizerAuthorizationStatus {
    let semaphore = DispatchSemaphore(value: 0)
    var authorizationStatus = SFSpeechRecognizerAuthorizationStatus.notDetermined
    SFSpeechRecognizer.requestAuthorization { status in
        authorizationStatus = status
        semaphore.signal()
    }
    let deadline = DispatchTime.now() + timeoutSec
    if semaphore.wait(timeout: deadline) == .timedOut {
        return .notDetermined
    }
    return authorizationStatus
}

let options = parseArgs()
let audioURL = URL(fileURLWithPath: options.audioPath)
let started = Date()

let authorizationStatus = requestSpeechAuthorization(timeoutSec: 60.0)
guard authorizationStatus == .authorized else {
    emitJSON([
        "ok": false,
        "text": "",
        "error": "speech_authorization_\(authorizationStatus.rawValue)",
        "locale": options.localeID,
        "on_device": options.onDevice,
        "infer_sec": Date().timeIntervalSince(started)
    ], outPath: options.outPath)
    exit(1)
}

guard let recognizer = SFSpeechRecognizer(locale: Locale(identifier: options.localeID)) else {
    emitJSON([
        "ok": false,
        "text": "",
        "error": "recognizer_unavailable_for_locale",
        "locale": options.localeID,
        "on_device": options.onDevice,
        "infer_sec": Date().timeIntervalSince(started)
    ], outPath: options.outPath)
    exit(1)
}

let request = SFSpeechURLRecognitionRequest(url: audioURL)
request.shouldReportPartialResults = false
if #available(macOS 10.15, *) {
    if options.onDevice && !recognizer.supportsOnDeviceRecognition {
        emitJSON([
            "ok": false,
            "text": "",
            "error": "on_device_not_supported_for_locale",
            "locale": options.localeID,
            "on_device": options.onDevice,
            "infer_sec": Date().timeIntervalSince(started)
        ], outPath: options.outPath)
        exit(1)
    }
    request.requiresOnDeviceRecognition = options.onDevice
}
if #available(macOS 13.0, *) {
    request.addsPunctuation = true
}

let lock = NSLock()
var completed = false
var finalText = ""
var errorText: String?

func finish() {
    lock.lock()
    let shouldSignal = !completed
    completed = true
    lock.unlock()
    _ = shouldSignal
}

let task = recognizer.recognitionTask(with: request) { result, error in
    if let result = result {
        finalText = result.bestTranscription.formattedString
        if result.isFinal {
            finish()
        }
    }
    if let error = error {
        errorText = error.localizedDescription
        finish()
    }
}

let deadline = Date().addingTimeInterval(options.timeoutSec)
while true {
    lock.lock()
    let isCompleted = completed
    lock.unlock()
    if isCompleted || Date() >= deadline {
        break
    }
    RunLoop.current.run(mode: .default, before: Date().addingTimeInterval(0.05))
}

lock.lock()
let isCompleted = completed
lock.unlock()
if !isCompleted {
    task.cancel()
    errorText = "timeout"
}

let elapsed = Date().timeIntervalSince(started)
let ok = errorText == nil && !finalText.isEmpty
emitJSON([
    "ok": ok,
    "text": finalText,
    "error": errorText as Any,
    "locale": options.localeID,
    "on_device": options.onDevice,
    "infer_sec": elapsed
], outPath: options.outPath)
exit(ok ? 0 : 1)
