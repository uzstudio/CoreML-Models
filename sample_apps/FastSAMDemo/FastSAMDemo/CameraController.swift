import AVFoundation
import CoreGraphics
import Combine
import SwiftUI
import SAMKit

/// Drives the camera and runs FastSAM "segment everything" on each frame.
///
/// Two big perf moves over a naïve pipeline:
///   • the preview is rendered by **AVCaptureVideoPreviewLayer** (zero CPU per frame) — we no
///     longer create a CGImage of every frame for display.
///   • inference takes the **CVPixelBuffer directly** (`FastSamSession.setImage(_ pb:)`) so
///     there is no CGImage round-trip, no per-pixel float-fill loop, and no CGContext draw —
///     the model's `ImageType` input does the resize + normalise on ANE/GPU.
///
/// `alwaysDiscardsLateVideoFrames` + a serial delegate queue already drops frames while one is
/// in flight, so the preview never backs up — no manual throttle needed.
final class CameraController: NSObject, ObservableObject {

    @Published var fps: Double = 0
    @Published var enabled = true
    @Published private(set) var isReady: Bool = false

    /// Overlay sink — set by `PreviewView` to push each mask straight onto a CALayer's
    /// `contents`, bypassing SwiftUI (the yolo-ios-app `maskLayer.contents = …` pattern).
    /// Routing a video-rate CGImage through `@Published` + `Image(uiImage:)` is what made the
    /// mask lag behind the smooth preview.
    var overlayHandler: ((CGImage?) -> Void)?

    /// Shared session — `PreviewView` attaches its preview layer to this.
    let session = AVCaptureSession()

    private let videoOutput = AVCaptureVideoDataOutput()
    private let queue = DispatchQueue(label: "fastsam.camera")
    private var fastSam: FastSamSession?
    private var lastStamp = CFAbsoluteTimeGetCurrent()
    private var modelName: String = "FastSAM_s_512"

    // Tunable from the UI (read on the camera queue, written from main — benign race).
    var confidence: Float = 0.5
    var maxInstances: Int = 40

    override init() {
        super.init()
        // FastSamSession.profile = true   // uncomment to print per-frame timing
        loadModel(name: modelName)
    }

    func loadModel(name: String) {
        queue.async { [weak self] in
            guard let self else { return }
            self.modelName = name
            do {
                let s = try FastSamSession(modelName: name,
                                           config: RuntimeConfig(computeUnits: .neuralEnginePreferred))
                s.trackColors = true   // stable per-object colours across frames
                self.fastSam = s
                DispatchQueue.main.async { self.isReady = true }
            } catch {
                print("[CameraController] FastSAM load failed: \(error)")
                self.fastSam = nil
                DispatchQueue.main.async { self.isReady = false }
            }
        }
    }

    func start() {
        AVCaptureDevice.requestAccess(for: .video) { [weak self] granted in
            guard granted else { return }
            self?.queue.async { self?.configureAndRun() }
        }
    }

    func stop() {
        queue.async { [weak self] in
            guard let self, self.session.isRunning else { return }
            self.session.stopRunning()
        }
    }

    /// Runs on `queue`. Idempotent — re-calls just start a stopped session.
    private func configureAndRun() {
        if !session.inputs.isEmpty {
            if !session.isRunning { session.startRunning() }
            return
        }
        session.beginConfiguration()
        session.sessionPreset = .hd1280x720
        if let device = AVCaptureDevice.default(.builtInWideAngleCamera, for: .video, position: .back),
           let input = try? AVCaptureDeviceInput(device: device),
           session.canAddInput(input) {
            session.addInput(input)
        }
        videoOutput.videoSettings = [kCVPixelBufferPixelFormatTypeKey as String: kCVPixelFormatType_32BGRA]
        videoOutput.alwaysDiscardsLateVideoFrames = true
        videoOutput.setSampleBufferDelegate(self, queue: queue)
        if session.canAddOutput(videoOutput) { session.addOutput(videoOutput) }
        if let conn = videoOutput.connection(with: .video), conn.isVideoOrientationSupported {
            conn.videoOrientation = .portrait
        }
        session.commitConfiguration()
        session.startRunning()
    }
}

extension CameraController: AVCaptureVideoDataOutputSampleBufferDelegate {
    func captureOutput(_ output: AVCaptureOutput, didOutput sampleBuffer: CMSampleBuffer,
                       from connection: AVCaptureConnection) {
        guard enabled, let fastSam,
              let pixelBuffer = CMSampleBufferGetImageBuffer(sampleBuffer) else { return }

        var newOverlay: CGImage?
        let opts = FastSamSession.Options(confidenceThreshold: confidence,
                                          iouThreshold: 0.9,
                                          maxInstances: maxInstances)
        do {
            try fastSam.setImage(pixelBuffer)
            newOverlay = try fastSam.segmentEverythingMask(options: opts)
        } catch {
            newOverlay = nil
        }

        let now = CFAbsoluteTimeGetCurrent()
        let dt = now - lastStamp
        lastStamp = now

        DispatchQueue.main.async {
            self.overlayHandler?(newOverlay)         // direct CALayer.contents — no SwiftUI churn
            if dt > 0 { self.fps = 1.0 / dt }
        }
    }
}

// MARK: - SwiftUI preview layer

/// Hosts an `AVCaptureVideoPreviewLayer` + a mask overlay `CALayer`. The system renders camera
/// frames directly (Core Animation), and each FastSAM mask is pushed onto `maskLayer.contents`
/// straight from the camera queue — SwiftUI never sees individual frames or masks.
struct PreviewView: UIViewRepresentable {
    let session: AVCaptureSession
    let controller: CameraController

    func makeUIView(context: Context) -> PreviewUIView {
        let v = PreviewUIView()
        v.previewLayer.session = session
        v.previewLayer.videoGravity = .resizeAspect
        if let conn = v.previewLayer.connection, conn.isVideoOrientationSupported {
            conn.videoOrientation = .portrait
        }
        controller.overlayHandler = { [weak v] cg in v?.setOverlay(cg) }
        return v
    }

    func updateUIView(_ uiView: PreviewUIView, context: Context) {}
}

final class PreviewUIView: UIView {
    override class var layerClass: AnyClass { AVCaptureVideoPreviewLayer.self }
    var previewLayer: AVCaptureVideoPreviewLayer { layer as! AVCaptureVideoPreviewLayer }

    private let maskLayer = CALayer()

    override init(frame: CGRect) {
        super.init(frame: frame)
        maskLayer.contentsGravity = .resizeAspect   // same fit as the .resizeAspect preview
        maskLayer.opacity = 0.55
        maskLayer.magnificationFilter = .nearest    // crisp instance edges from the small mask
        layer.addSublayer(maskLayer)
    }
    required init?(coder: NSCoder) { fatalError("init(coder:) has not been implemented") }

    override func layoutSubviews() {
        super.layoutSubviews()
        maskLayer.frame = bounds
    }

    /// Push a new mask image (cheap GPU texture swap; implicit animation disabled).
    func setOverlay(_ cg: CGImage?) {
        CATransaction.begin()
        CATransaction.setDisableActions(true)
        maskLayer.contents = cg
        CATransaction.commit()
    }
}
