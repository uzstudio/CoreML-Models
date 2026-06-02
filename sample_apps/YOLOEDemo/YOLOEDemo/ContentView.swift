import SwiftUI
import UIKit
import AVFoundation
import CoreML
import Vision
import Accelerate
import PhotosUI
import AVKit

// MARK: - Main TabView

struct ContentView: View {
    @StateObject private var detector = TextGroundingDetector()
    @State private var selectedTab = 2
    @State private var threshold: Float = 0.15

    var body: some View {
        ZStack {
            TabView(selection: $selectedTab) {
                PhotoDetectionView(detector: detector, threshold: $threshold)
                    .tabItem { Label("Photo", systemImage: "photo") }
                    .tag(0)
                VideoDetectionView(detector: detector, threshold: $threshold)
                    .tabItem { Label("Video", systemImage: "video") }
                    .tag(1)
                CameraDetectionView(detector: detector, threshold: $threshold)
                    .tabItem { Label("Camera", systemImage: "camera") }
                    .tag(2)
                VisualDetectionView(detector: detector)
                    .tabItem { Label("Visual", systemImage: "viewfinder") }
                    .tag(3)
            }

            if !detector.isModelLoaded {
                Color.black.ignoresSafeArea()
                VStack(spacing: 16) {
                    ProgressView().tint(.white).scaleEffect(1.2)
                    Text("Loading model...").font(.subheadline).foregroundColor(.secondary)
                }
            }
        }
        .tint(.white)
        .preferredColorScheme(.dark)
    }
}

// MARK: - Detection Result

struct Detection: Identifiable {
    let id = UUID()
    let label: String
    let confidence: Float
    let classIndex: Int
    let normRect: CGRect // Normalized [0,1], origin at top-left
    let anchorIndex: Int // Anchor index for mask lookup
}

// MARK: - Detection Overlay (shared for photo & video)

struct DetectionOverlay: View {
    let detections: [Detection]
    let imageSize: CGSize
    let displaySize: CGSize
    let colors: [UIColor]

    var body: some View {
        let transform = aspectFitTransform()
        ForEach(detections) { det in
            let r = scaledRect(det.normRect, transform: transform)
            let color = Color(colors[det.classIndex % colors.count])

            RoundedRectangle(cornerRadius: 10)
                .stroke(color, lineWidth: 2)
                .background(RoundedRectangle(cornerRadius: 10).fill(color.opacity(0.08)))
                .frame(width: r.width, height: r.height)
                .position(x: r.midX, y: r.midY)

            Text("  \(det.label) \(Int(det.confidence * 100))%  ")
                .font(.system(size: 11, weight: .semibold))
                .foregroundColor(.white)
                .padding(.horizontal, 4)
                .padding(.vertical, 2)
                .background(color.opacity(0.85))
                .cornerRadius(8)
                .position(x: r.midX, y: r.minY > 20 ? r.minY - 14 : r.maxY + 14)
        }
    }

    private struct FitTransform { let scale: CGFloat; let offsetX: CGFloat; let offsetY: CGFloat }

    private func aspectFitTransform() -> FitTransform {
        guard imageSize.width > 0, imageSize.height > 0 else {
            return FitTransform(scale: 1, offsetX: 0, offsetY: 0)
        }
        let scaleX = displaySize.width / imageSize.width
        let scaleY = displaySize.height / imageSize.height
        let scale = min(scaleX, scaleY)
        let scaledW = imageSize.width * scale
        let scaledH = imageSize.height * scale
        return FitTransform(scale: scale,
                            offsetX: (displaySize.width - scaledW) / 2,
                            offsetY: (displaySize.height - scaledH) / 2)
    }

    private func scaledRect(_ nr: CGRect, transform t: FitTransform) -> CGRect {
        CGRect(x: nr.minX * imageSize.width * t.scale + t.offsetX,
               y: nr.minY * imageSize.height * t.scale + t.offsetY,
               width: nr.width * imageSize.width * t.scale,
               height: nr.height * imageSize.height * t.scale)
    }
}

// MARK: - Photo Detection

struct PhotoDetectionView: View {
    let detector: TextGroundingDetector
    @Binding var threshold: Float
    @State private var selectedItem: PhotosPickerItem?
    @State private var image: UIImage?
    @State private var overlayImage: UIImage?
    @State private var allDetections: [Detection] = []
    @State private var isProcessing = false
    @State private var inferenceTime: Double = 0
    @State private var queryText = "person, dog, car"
    @State private var maxDet = 50   // shared detection cap (applies to all modes via the detector)

    private var filteredDetections: [Detection] {
        allDetections.filter { $0.confidence >= threshold }
    }

    var body: some View {
        ZStack {
            Color.black.ignoresSafeArea()

            if let displayImage = overlayImage ?? image {
                Image(uiImage: displayImage)
                    .resizable()
                    .aspectRatio(contentMode: .fit)
                    .frame(maxWidth: .infinity, maxHeight: .infinity)
            } else {
                PhotosPicker(selection: $selectedItem, matching: .images) {
                    VStack(spacing: 12) {
                        Image(systemName: "photo.on.rectangle.angled")
                            .font(.system(size: 48))
                        Text("Tap to select a photo").font(.subheadline)
                    }
                    .foregroundStyle(.secondary)
                }
            }

            VStack {
                Spacer()
                VStack(spacing: 8) {
                    HStack(spacing: 8) {
                        TextField("Objects (comma-separated)", text: $queryText)
                            .textFieldStyle(.roundedBorder)
                            .onSubmit { runDetection() }
                            .submitLabel(.search)
                        Button { runDetection() } label: {
                            Image(systemName: "magnifyingglass")
                                .foregroundColor(.white)
                                .padding(6)
                                .background(Color.blue, in: Circle())
                        }
                    }
                    HStack(spacing: 4) {
                        Text(String(format: "%.0f%%", threshold * 100))
                            .font(.caption).monospacedDigit().frame(width: 36)
                        Slider(value: $threshold, in: 0.05...0.95, step: 0.05)
                            .onChange(of: threshold) { val in
                                detector.confidenceThreshold = val
                                updateOverlay()
                            }
                        Menu {
                            Picker("Max detections", selection: $maxDet) {
                                ForEach([25, 50, 75, 100], id: \.self) { Text("\($0)").tag($0) }
                            }
                        } label: {
                            Text("≤\(maxDet)").font(.caption).monospacedDigit().foregroundColor(.white)
                                .padding(.horizontal, 7).padding(.vertical, 3)
                                .background(.ultraThinMaterial, in: Capsule())
                        }
                        .onChange(of: maxDet) { val in
                            detector.maxDetections = val
                            if image != nil { runDetection() }
                        }
                    }
                    HStack {
                        if !filteredDetections.isEmpty {
                            Text("\(filteredDetections.count) objects").font(.caption)
                        }
                        Spacer()
                        if inferenceTime > 0 {
                            Text(String(format: "%.0fms", inferenceTime))
                                .font(.caption).foregroundStyle(.secondary)
                        }
                        PhotosPicker(selection: $selectedItem, matching: .images) {
                            Image(systemName: "photo.badge.plus")
                                .font(.body).foregroundColor(.white)
                        }
                    }
                    .foregroundColor(.white)
                }
                .padding(.horizontal, 20)
                .padding(.vertical, 10)
                .background(.ultraThinMaterial)
            }

            if isProcessing {
                ProgressView().tint(.white).scaleEffect(1.5)
                    .padding(24)
                    .background(.ultraThinMaterial, in: RoundedRectangle(cornerRadius: 12))
            }
        }
        .onChange(of: selectedItem) { _ in loadAndDetect() }
        .onTapGesture { UIApplication.shared.sendAction(#selector(UIResponder.resignFirstResponder), to: nil, from: nil, for: nil) }
    }

    private func loadAndDetect() {
        guard let selectedItem else { return }
        isProcessing = true
        Task {
            if let data = try? await selectedItem.loadTransferable(type: Data.self),
               let uiImage = UIImage(data: data) {
                detector.updateQueries(queryText)
                let start = CFAbsoluteTimeGetCurrent()
                let result = detector.detectSyncWithMasks(image: uiImage)
                let elapsed = (CFAbsoluteTimeGetCurrent() - start) * 1000
                let rendered = renderMaskOverlay(on: uiImage, detections: result.detections, maskData: result.maskData)
                await MainActor.run {
                    image = uiImage
                    allDetections = result.detections
                    overlayImage = rendered
                    inferenceTime = elapsed
                    isProcessing = false
                }
            } else {
                await MainActor.run { isProcessing = false }
            }
        }
    }

    private func runDetection() {
        guard let image else { return }
        isProcessing = true
        detector.updateQueries(queryText)
        Task {
            let start = CFAbsoluteTimeGetCurrent()
            let result = detector.detectSyncWithMasks(image: image)
            let elapsed = (CFAbsoluteTimeGetCurrent() - start) * 1000
            let rendered = renderMaskOverlay(on: image, detections: result.detections, maskData: result.maskData)
            await MainActor.run {
                allDetections = result.detections
                overlayImage = rendered
                inferenceTime = elapsed
                isProcessing = false
            }
        }
    }

    private func updateOverlay() {
        guard let image else { return }
        let dets = filteredDetections
        guard !dets.isEmpty, let maskData = detector.lastMaskData else {
            overlayImage = image
            return
        }
        Task {
            let rendered = renderMaskOverlay(on: image, detections: dets, maskData: maskData)
            await MainActor.run { overlayImage = rendered }
        }
    }

    /// Render bounding boxes and semi-transparent colored masks on top of the original image
    private func renderMaskOverlay(on image: UIImage, detections: [Detection], maskData: MaskData?) -> UIImage {
        guard !detections.isEmpty else { return image }

        let size = image.size
        let renderer = UIGraphicsImageRenderer(size: size)
        return renderer.image { ctx in
            image.draw(in: CGRect(origin: .zero, size: size))

            // Draw bounding boxes
            for det in detections {
                guard det.confidence >= detector.confidenceThreshold else { continue }
                let color = detector.colors[det.classIndex % detector.colors.count]
                let boxRect = CGRect(
                    x: CGFloat(det.normRect.minX) * size.width,
                    y: CGFloat(det.normRect.minY) * size.height,
                    width: CGFloat(det.normRect.width) * size.width,
                    height: CGFloat(det.normRect.height) * size.height
                )
                ctx.cgContext.setStrokeColor(color.cgColor)
                ctx.cgContext.setLineWidth(max(size.width / 200, 2))
                ctx.cgContext.stroke(boxRect)

                // Label
                let label = "\(det.label) \(Int(det.confidence * 100))%"
                let attrs: [NSAttributedString.Key: Any] = [
                    .font: UIFont.boldSystemFont(ofSize: max(size.width / 40, 14)),
                    .foregroundColor: UIColor.white,
                    .backgroundColor: color.withAlphaComponent(0.8)
                ]
                let labelStr = NSAttributedString(string: " \(label) ", attributes: attrs)
                labelStr.draw(at: CGPoint(x: boxRect.minX, y: max(boxRect.minY - max(size.width / 30, 20), 0)))
            }

            guard let maskData = maskData else { return }

            let maskProtoSize = 160
            let detectorInputSize = 640

            for det in detections {
                guard det.confidence >= detector.confidenceThreshold else { continue }
                let anchorIdx = det.anchorIndex
                guard anchorIdx < maskData.numAnchors else { continue }

                // Compute mask: sigmoid(coeffs @ protos) -> 160x160
                var mask160 = [Float](repeating: 0, count: maskProtoSize * maskProtoSize)
                for k in 0..<32 {
                    let coeff = maskData.coeffs[k * maskData.numAnchors + anchorIdx]
                    let protoOffset = k * maskProtoSize * maskProtoSize
                    for p in 0..<(maskProtoSize * maskProtoSize) {
                        mask160[p] += coeff * maskData.protos[protoOffset + p]
                    }
                }
                // Sigmoid
                for p in 0..<mask160.count {
                    mask160[p] = 1.0 / (1.0 + exp(-mask160[p]))
                }

                // The detection normRect is in original image coordinates [0,1].
                // The mask160 corresponds to the 640x640 padded input.
                // We need to figure out the pad/scale used during preprocessing to
                // map from original image coords to 640x640 coords, then sample the mask.

                let imgW = Float(image.size.width)
                let imgH = Float(image.size.height)
                let scale = Float(detectorInputSize) / max(imgW, imgH)
                let scaledW = imgW * scale
                let scaledH = imgH * scale
                let padX = (Float(detectorInputSize) - scaledW) / 2
                let padY = (Float(detectorInputSize) - scaledH) / 2

                // Bounding box in original image pixels
                let boxX = Float(det.normRect.minX) * imgW
                let boxY = Float(det.normRect.minY) * imgH
                let boxW = Float(det.normRect.width) * imgW
                let boxH = Float(det.normRect.height) * imgH

                guard boxW > 0, boxH > 0 else { continue }

                let pixW = Int(ceil(boxW))
                let pixH = Int(ceil(boxH))
                guard pixW > 0, pixH > 0 else { continue }

                // For each pixel in the bounding box region, sample from the 160x160 mask
                // Mapping: origPixel -> 640 coord -> 160 coord
                // 640_coord = origPixel * scale + pad
                // 160_coord = 640_coord * (160/640) = 640_coord / 4
                let maskScale: Float = Float(maskProtoSize) / Float(detectorInputSize)

                var maskPixels = [UInt8](repeating: 0, count: pixW * pixH * 4)
                let color = detector.colors[det.classIndex % detector.colors.count]
                var r: CGFloat = 0, g: CGFloat = 0, b: CGFloat = 0, a: CGFloat = 0
                color.getRed(&r, green: &g, blue: &b, alpha: &a)
                let cr = UInt8(r * 255)
                let cg = UInt8(g * 255)
                let cb = UInt8(b * 255)

                for py in 0..<pixH {
                    for px in 0..<pixW {
                        let origX = boxX + Float(px)
                        let origY = boxY + Float(py)

                        // Map to 160x160 mask space
                        let mx = (origX * scale + padX) * maskScale
                        let my = (origY * scale + padY) * maskScale

                        let ix = Int(mx)
                        let iy = Int(my)
                        guard ix >= 0, ix < maskProtoSize, iy >= 0, iy < maskProtoSize else { continue }

                        let maskVal = mask160[iy * maskProtoSize + ix]
                        if maskVal > 0.5 {
                            let alpha = UInt8(min(maskVal * 0.5, 1.0) * 255)
                            let offset = (py * pixW + px) * 4
                            maskPixels[offset + 0] = cr
                            maskPixels[offset + 1] = cg
                            maskPixels[offset + 2] = cb
                            maskPixels[offset + 3] = alpha
                        }
                    }
                }

                // Draw mask region
                if let maskCGImage = createCGImage(from: maskPixels, width: pixW, height: pixH) {
                    let drawRect = CGRect(x: CGFloat(boxX), y: CGFloat(boxY),
                                          width: CGFloat(pixW), height: CGFloat(pixH))
                    ctx.cgContext.draw(maskCGImage, in: drawRect)
                }
            }
        }
    }

    private func createCGImage(from pixels: [UInt8], width: Int, height: Int) -> CGImage? {
        guard width > 0, height > 0 else { return nil }
        let colorSpace = CGColorSpaceCreateDeviceRGB()
        let bitmapInfo = CGBitmapInfo(rawValue: CGImageAlphaInfo.premultipliedLast.rawValue)
        guard let provider = CGDataProvider(data: Data(pixels) as CFData) else { return nil }
        return CGImage(width: width, height: height, bitsPerComponent: 8, bitsPerPixel: 32,
                       bytesPerRow: width * 4, space: colorSpace, bitmapInfo: bitmapInfo,
                       provider: provider, decode: nil, shouldInterpolate: true,
                       intent: .defaultIntent)
    }
}

// MARK: - Video Detection

struct VideoDetectionView: View {
    let detector: TextGroundingDetector
    @Binding var threshold: Float
    @State private var selectedItem: PhotosPickerItem?
    @State private var currentFrame: UIImage?
    @State private var currentMask: CGImage?
    @State private var detections: [Detection] = []
    @State private var progress: Double = 0
    @State private var fps: Double = 0
    @State private var playbackTask: Task<Void, Never>?
    @State private var queryText = "person, dog, car"

    var body: some View {
        ZStack {
            Color.black.ignoresSafeArea()

            if let currentFrame {
                GeometryReader { geo in
                    Image(uiImage: currentFrame)
                        .resizable()
                        .aspectRatio(contentMode: .fit)
                        .frame(maxWidth: .infinity, maxHeight: .infinity)

                    // Segmentation overlay: same aspect-fit as the frame -> aligns with boxes.
                    if let currentMask {
                        Image(decorative: currentMask, scale: 1)
                            .resizable()
                            .interpolation(.none)
                            .aspectRatio(contentMode: .fit)
                            .frame(maxWidth: .infinity, maxHeight: .infinity)
                            .opacity(0.5)
                            .allowsHitTesting(false)
                    }

                    DetectionOverlay(detections: detections,
                                     imageSize: currentFrame.size,
                                     displaySize: geo.size,
                                     colors: detector.colors)
                }
            } else {
                PhotosPicker(selection: $selectedItem, matching: .videos) {
                    VStack(spacing: 12) {
                        Image(systemName: "video.badge.plus")
                            .font(.system(size: 48))
                        Text("Tap to select a video").font(.subheadline)
                    }
                    .foregroundStyle(.secondary)
                }
            }

            VStack {
                Spacer()
                VStack(spacing: 8) {
                    if currentFrame != nil {
                        ProgressView(value: progress).tint(.blue)
                    }
                    HStack(spacing: 8) {
                        TextField("Objects (comma-separated)", text: $queryText)
                            .textFieldStyle(.roundedBorder)
                            .onSubmit { detector.updateQueries(queryText) }
                            .submitLabel(.search)
                        Button { detector.updateQueries(queryText) } label: {
                            Image(systemName: "magnifyingglass")
                                .foregroundColor(.white)
                                .padding(6)
                                .background(Color.blue, in: Circle())
                        }
                    }
                    HStack(spacing: 4) {
                        Text(String(format: "%.0f%%", threshold * 100))
                            .font(.caption).monospacedDigit().frame(width: 36)
                        Slider(value: $threshold, in: 0.05...0.95, step: 0.05)
                            .onChange(of: threshold) { val in detector.confidenceThreshold = val }
                    }
                    HStack {
                        if !detections.isEmpty {
                            Text("\(detections.count) objects").font(.caption)
                        }
                        Spacer()
                        if fps > 0 {
                            Text(String(format: "%.1f FPS", fps))
                                .font(.caption).foregroundStyle(.secondary)
                        }
                        PhotosPicker(selection: $selectedItem, matching: .videos) {
                            Image(systemName: "video.badge.plus")
                                .font(.body).foregroundColor(.white)
                        }
                    }
                    .foregroundColor(.white)
                }
                .padding(.horizontal, 20)
                .padding(.vertical, 10)
                .background(.ultraThinMaterial)
            }
        }
        .onChange(of: selectedItem) { _ in loadAndProcess() }
        .onDisappear { playbackTask?.cancel() }
        .onTapGesture { UIApplication.shared.sendAction(#selector(UIResponder.resignFirstResponder), to: nil, from: nil, for: nil) }
    }

    private func loadAndProcess() {
        playbackTask?.cancel()
        guard let selectedItem else { return }
        detector.updateQueries(queryText)

        Task {
            guard let videoData = try? await selectedItem.loadTransferable(type: VideoTransferable.self) else { return }
            playbackTask = Task.detached(priority: .userInitiated) {
                await processVideo(url: videoData.url)
            }
        }
    }

    private func processVideo(url: URL) async {
        let asset = AVURLAsset(url: url)
        guard let track = try? await asset.loadTracks(withMediaType: .video).first else { return }
        let duration = try? await asset.load(.duration)
        let totalSeconds = duration.map { CMTimeGetSeconds($0) } ?? 1

        guard let reader = try? AVAssetReader(asset: asset) else { return }
        let outputSettings: [String: Any] = [kCVPixelBufferPixelFormatTypeKey as String: kCVPixelFormatType_32BGRA]
        let trackOutput = AVAssetReaderTrackOutput(track: track, outputSettings: outputSettings)
        reader.add(trackOutput)
        reader.startReading()

        let ciContext = CIContext()
        let nominalFPS = (try? await track.load(.nominalFrameRate)) ?? 30
        let frameInterval = 1.0 / Double(nominalFPS)

        while !Task.isCancelled, let sb = trackOutput.copyNextSampleBuffer() {
            let pts = CMSampleBufferGetPresentationTimeStamp(sb)
            let currentSec = CMTimeGetSeconds(pts)

            guard let pb = CMSampleBufferGetImageBuffer(sb) else { continue }
            let ciImage = CIImage(cvPixelBuffer: pb)
            guard let cgImage = ciContext.createCGImage(ciImage, from: ciImage.extent) else { continue }
            let frame = UIImage(cgImage: cgImage)

            let start = CFAbsoluteTimeGetCurrent()
            let result = detector.detectSync(image: frame)
            let elapsed = CFAbsoluteTimeGetCurrent() - start
            let currentFPS = 1.0 / max(elapsed, 0.001)

            await MainActor.run {
                currentFrame = frame
                detections = result.detections
                currentMask = result.maskImage
                progress = currentSec / totalSeconds
                fps = fps == 0 ? currentFPS : fps * 0.9 + currentFPS * 0.1
            }

            let sleepTime = max(frameInterval - elapsed, 0)
            if sleepTime > 0 { try? await Task.sleep(for: .seconds(sleepTime)) }
        }

        await MainActor.run { progress = 1.0 }
    }
}

struct VideoTransferable: Transferable {
    let url: URL
    static var transferRepresentation: some TransferRepresentation {
        FileRepresentation(contentType: .movie) { movie in
            SentTransferredFile(movie.url)
        } importing: { received in
            let tmp = FileManager.default.temporaryDirectory
                .appendingPathComponent(UUID().uuidString + "." + received.file.pathExtension)
            try FileManager.default.copyItem(at: received.file, to: tmp)
            return Self(url: tmp)
        }
    }
}

// MARK: - Camera Detection (UIKit -- CALayer pool)

struct CameraDetectionView: View {
    let detector: TextGroundingDetector
    @Binding var threshold: Float
    var showsQueryUI: Bool = true   // false when driven by an external (e.g. visual) query
    var body: some View {
        CameraVCWrapper(detector: detector, threshold: $threshold, showsQueryUI: showsQueryUI)
            .ignoresSafeArea(edges: .bottom)
    }
}

struct CameraVCWrapper: UIViewControllerRepresentable {
    let detector: TextGroundingDetector
    @Binding var threshold: Float
    var showsQueryUI: Bool = true
    func makeUIViewController(context: Context) -> CameraVC { CameraVC(detector: detector, showsQueryUI: showsQueryUI) }
    func updateUIViewController(_ vc: CameraVC, context: Context) {
        detector.confidenceThreshold = threshold
    }
}

class CameraVC: UIViewController, AVCaptureVideoDataOutputSampleBufferDelegate {
    private let detector: TextGroundingDetector
    private let session = AVCaptureSession()
    private let sessionQueue = DispatchQueue(label: "session")
    private let inferenceQueue = DispatchQueue(label: "inference")
    private var previewLayer: AVCaptureVideoPreviewLayer!
    private let maskLayer = CALayer()
    private var boxViews: [BoundingBoxView] = []
    private var isProcessing = false
    private var longSide: CGFloat = 1920
    private var shortSide: CGFloat = 1080
    private var frameSizeCaptured = false

    private var smoothedMs: Double = 0
    private var smoothedFps: Double = 0
    private let statsLabel = CATextLayer()

    // Query UI
    private let queryField = UITextField()
    private let queryBar = UIVisualEffectView(effect: UIBlurEffect(style: .dark))
    private var queryBarBottom: NSLayoutConstraint?
    private let showsQueryUI: Bool

    init(detector: TextGroundingDetector, showsQueryUI: Bool = true) {
        self.detector = detector
        self.showsQueryUI = showsQueryUI
        super.init(nibName: nil, bundle: nil)
    }
    required init?(coder: NSCoder) { fatalError() }

    override func viewDidLoad() {
        super.viewDidLoad()
        view.backgroundColor = .black

        previewLayer = AVCaptureVideoPreviewLayer(session: session)
        previewLayer.videoGravity = .resizeAspectFill
        view.layer.addSublayer(previewLayer)

        // Segmentation overlay: a small proto-res mask CGImage scaled up by Core Animation
        // (yolo-ios-app maskLayer.contents pattern). Same gravity as the preview so it aligns.
        maskLayer.contentsGravity = .resizeAspectFill
        maskLayer.opacity = 0.5
        maskLayer.magnificationFilter = .nearest   // crisp instance edges from the 160² mask
        previewLayer.addSublayer(maskLayer)

        for _ in 0..<100 {
            let bv = BoundingBoxView()
            bv.addToLayer(previewLayer)
            boxViews.append(bv)
        }

        // Stats overlay
        statsLabel.fontSize = 13
        statsLabel.font = UIFont.monospacedSystemFont(ofSize: 13, weight: .medium)
        statsLabel.foregroundColor = UIColor.white.cgColor
        statsLabel.backgroundColor = UIColor.black.withAlphaComponent(0.5).cgColor
        statsLabel.cornerRadius = 8
        statsLabel.masksToBounds = true
        statsLabel.contentsScale = UIScreen.main.scale
        statsLabel.alignmentMode = .center
        view.layer.addSublayer(statsLabel)

        // Query bar (text mode only; in visual mode the query is set externally)
        if showsQueryUI {
            queryBar.layer.cornerRadius = 12
            queryBar.clipsToBounds = true
            queryBar.translatesAutoresizingMaskIntoConstraints = false
            view.addSubview(queryBar)

            queryField.placeholder = "Objects (comma-separated)"
            queryField.text = "person, dog, car"
            queryField.borderStyle = .roundedRect
            queryField.font = .systemFont(ofSize: 14)
            queryField.returnKeyType = .search
            queryField.autocorrectionType = .no
            queryField.delegate = self
            queryField.translatesAutoresizingMaskIntoConstraints = false
            queryBar.contentView.addSubview(queryField)

            let barBottom = queryBar.bottomAnchor.constraint(equalTo: view.safeAreaLayoutGuide.bottomAnchor, constant: -8)
            queryBarBottom = barBottom
            NSLayoutConstraint.activate([
                queryBar.leadingAnchor.constraint(equalTo: view.leadingAnchor, constant: 12),
                queryBar.trailingAnchor.constraint(equalTo: view.trailingAnchor, constant: -12),
                barBottom,
                queryBar.heightAnchor.constraint(equalToConstant: 48),
                queryField.leadingAnchor.constraint(equalTo: queryBar.contentView.leadingAnchor, constant: 8),
                queryField.trailingAnchor.constraint(equalTo: queryBar.contentView.trailingAnchor, constant: -8),
                queryField.centerYAnchor.constraint(equalTo: queryBar.contentView.centerYAnchor),
            ])

            // Lift the query bar above the keyboard; tap the preview to dismiss it.
            let nc = NotificationCenter.default
            nc.addObserver(self, selector: #selector(keyboardWillChange(_:)), name: UIResponder.keyboardWillShowNotification, object: nil)
            nc.addObserver(self, selector: #selector(keyboardWillChange(_:)), name: UIResponder.keyboardWillHideNotification, object: nil)
            let tap = UITapGestureRecognizer(target: self, action: #selector(dismissKeyboard))
            tap.cancelsTouchesInView = false
            view.addGestureRecognizer(tap)

            detector.updateQueries(queryField.text ?? "")
        }

        AVCaptureDevice.requestAccess(for: .video) { [weak self] ok in
            guard ok else { return }
            self?.sessionQueue.async { self?.setupCamera() }
        }
    }

    @objc private func dismissKeyboard() { view.endEditing(true) }

    @objc private func keyboardWillChange(_ note: Notification) {
        guard let info = note.userInfo,
              let end = (info[UIResponder.keyboardFrameEndUserInfoKey] as? NSValue)?.cgRectValue else { return }
        let showing = note.name == UIResponder.keyboardWillShowNotification
        // Bar sits 8pt above the keyboard while shown, else 8pt above the safe-area bottom.
        let lift = showing ? (end.height - view.safeAreaInsets.bottom) : 0
        queryBarBottom?.constant = -8 - max(0, lift)
        let duration = (info[UIResponder.keyboardAnimationDurationUserInfoKey] as? Double) ?? 0.25
        let curveRaw = (info[UIResponder.keyboardAnimationCurveUserInfoKey] as? UInt) ?? UInt(UIView.AnimationCurve.easeInOut.rawValue)
        UIView.animate(withDuration: duration, delay: 0,
                       options: UIView.AnimationOptions(rawValue: curveRaw << 16)) {
            self.view.layoutIfNeeded()
        }
    }

    override func viewDidLayoutSubviews() {
        super.viewDidLayoutSubviews()
        previewLayer.frame = view.bounds
        CATransaction.begin(); CATransaction.setDisableActions(true)
        maskLayer.frame = previewLayer.bounds
        CATransaction.commit()
        statsLabel.frame = CGRect(
            x: (view.bounds.width - 220) / 2,
            y: view.safeAreaInsets.top + 8,
            width: 220, height: 28
        )
    }

    override func viewWillAppear(_ animated: Bool) {
        super.viewWillAppear(animated)
        sessionQueue.async { if !self.session.isRunning { self.session.startRunning() } }
    }

    override func viewWillDisappear(_ animated: Bool) {
        super.viewWillDisappear(animated)
        sessionQueue.async { self.session.stopRunning() }
    }

    private func setupCamera() {
        session.beginConfiguration()
        session.sessionPreset = .high
        guard let dev = AVCaptureDevice.default(.builtInWideAngleCamera, for: .video, position: .back),
              let input = try? AVCaptureDeviceInput(device: dev) else { session.commitConfiguration(); return }
        if session.canAddInput(input) { session.addInput(input) }
        let out = AVCaptureVideoDataOutput()
        out.videoSettings = [kCVPixelBufferPixelFormatTypeKey as String: kCVPixelFormatType_32BGRA]
        out.alwaysDiscardsLateVideoFrames = true
        out.setSampleBufferDelegate(self, queue: inferenceQueue)
        if session.canAddOutput(out) { session.addOutput(out) }
        session.commitConfiguration()
        out.connection(with: .video)?.videoOrientation = .portrait
        previewLayer.connection?.videoOrientation = .portrait
        session.startRunning()
    }

    func captureOutput(_ output: AVCaptureOutput, didOutput sb: CMSampleBuffer, from conn: AVCaptureConnection) {
        guard !isProcessing else { return }
        guard let pb = CMSampleBufferGetImageBuffer(sb) else { return }
        if !frameSizeCaptured {
            let w = CGFloat(CVPixelBufferGetWidth(pb))
            let h = CGFloat(CVPixelBufferGetHeight(pb))
            longSide = max(w, h); shortSide = min(w, h)
            frameSizeCaptured = true
        }
        isProcessing = true
        let start = CACurrentMediaTime()
        let result = detector.detectSync(pixelBuffer: pb)
        let dets = result.detections
        let maskCG = result.maskImage
        let ms = (CACurrentMediaTime() - start) * 1000
        isProcessing = false

        smoothedMs = smoothedMs == 0 ? ms : smoothedMs * 0.8 + ms * 0.2
        smoothedFps = smoothedFps == 0 ? 1000/ms : smoothedFps * 0.8 + (1000/ms) * 0.2

        // Convert to Vision-style rects for display
        let visionDets = dets.map { d in
            (d.label, d.confidence, d.classIndex,
             CGRect(x: d.normRect.minX, y: 1 - d.normRect.maxY,
                    width: d.normRect.width, height: d.normRect.height))
        }
        let statsText = String(format: "  %.1f ms  |  %.1f FPS  ", smoothedMs, smoothedFps)

        DispatchQueue.main.async {
            self.showBoxes(visionDets)
            CATransaction.begin()
            CATransaction.setDisableActions(true)
            self.maskLayer.contents = maskCG       // proto-res overlay, scaled by Core Animation
            self.statsLabel.string = statsText
            CATransaction.commit()
        }
    }

    private func showBoxes(_ dets: [(String, Float, Int, CGRect)]) {
        let width = view.bounds.width
        let height = view.bounds.height
        let ratio = (height / width) / (longSide / shortSide)

        for i in 0..<boxViews.count {
            guard i < dets.count else { boxViews[i].hide(); continue }   // dets already capped at maxDetections (<=100 = pool size)
            let (label, conf, cid, nr) = dets[i]
            var displayRect = nr

            if ratio >= 1 {
                let offset = (1 - ratio) * (0.5 - displayRect.minX)
                let transform = CGAffineTransform(scaleX: 1, y: -1).translatedBy(x: offset, y: -1)
                displayRect = displayRect.applying(transform)
                displayRect.size.width *= ratio
            } else {
                let offset = (ratio - 1) * (0.5 - displayRect.maxY)
                let transform = CGAffineTransform(scaleX: 1, y: -1).translatedBy(x: 0, y: offset - 1)
                displayRect = displayRect.applying(transform)
                let r2 = (height / width) / (shortSide / longSide)
                displayRect.size.height /= r2
            }

            let screenRect = VNImageRectForNormalizedRect(displayRect, Int(width), Int(height))
            let color = detector.colors[cid % detector.colors.count]
            let text = String(format: "%@ %.0f%%", label, conf * 100)
            // Keep boxes solid and legible (only above-threshold detections reach here);
            // a faint confidence cue remains in the 0.9-1.0 range.
            let alpha = CGFloat(max(0.9, min(1.0, conf)))
            boxViews[i].show(frame: screenRect, label: text, color: color, alpha: alpha)
        }
    }
}

extension CameraVC: UITextFieldDelegate {
    func textFieldShouldReturn(_ textField: UITextField) -> Bool {
        textField.resignFirstResponder()
        detector.updateQueries(textField.text ?? "")
        return true
    }
}

// MARK: - Visual Prompt Detection (detect "the same object" as a reference region)

struct VisualDetectionView: View {
    @ObservedObject var detector: TextGroundingDetector
    // Visual matches run weaker than text, so the Visual tab keeps its own lower default
    // threshold (the slider still adjusts it) rather than sharing the text tabs' 0.15.
    @State private var threshold: Float = 0.1

    @State private var selectedItem: PhotosPickerItem?
    @State private var refImage: UIImage?
    @State private var dragRect: CGRect = .zero        // live drag, container coords
    @State private var activeMask80: [Float]?          // selected reference mask (tap or box)
    @State private var selectionOverlay: CGImage?      // de-letterboxed mask overlay for feedback
    @State private var isSelecting = false
    @State private var queryActive = false
    @State private var showCamera = false

    private var cameraAvailable: Bool { UIImagePickerController.isSourceTypeAvailable(.camera) }

    var body: some View {
        ZStack {
            Color.black.ignoresSafeArea()
            if queryActive { liveCamera } else { selection }
        }
        .onChange(of: selectedItem) { _ in loadRef() }
        .fullScreenCover(isPresented: $showCamera) {
            CameraPicker { img in refImage = uprightImage(img); clearSelection() }
                .ignoresSafeArea()
        }
    }

    // Live camera detecting the visual query (no text field).
    private var liveCamera: some View {
        ZStack(alignment: .top) {
            CameraDetectionView(detector: detector, threshold: $threshold, showsQueryUI: false)
            VStack(spacing: 8) {
                HStack(spacing: 10) {
                    Text(String(format: "Best match: %.0f%%", detector.lastMaxScore * 100))
                        .font(.caption.bold()).foregroundColor(.white)
                    Spacer()
                    Button { queryActive = false } label: {
                        Label("Change", systemImage: "arrow.uturn.backward").font(.caption.bold())
                            .foregroundColor(.white).padding(.horizontal, 10).padding(.vertical, 6)
                            .background(Color.blue, in: Capsule())
                    }
                }
                // Visual matches are often weaker than text; let the user drop the threshold.
                HStack(spacing: 6) {
                    Image(systemName: "dial.min").font(.caption2).foregroundColor(.white.opacity(0.7))
                    Slider(value: $threshold, in: 0.02...0.8, step: 0.01)
                        .onChange(of: threshold) { val in detector.confidenceThreshold = val }
                    Text(String(format: "%.0f%%", threshold * 100))
                        .font(.caption2).monospacedDigit().foregroundColor(.white).frame(width: 32)
                }
            }
            .padding(.horizontal, 14).padding(.vertical, 10)
            .background(.ultraThinMaterial)
            .padding(.top, 50)
        }
        .onAppear { detector.confidenceThreshold = threshold; applyVisualQuery() }
    }

    // Pick a reference image; tap an object (auto-segment) or drag a box.
    private var selection: some View {
        VStack(spacing: 0) {
            if let refImage {
                GeometryReader { geo in
                    let fit = fittedRect(refImage.size, geo.size)
                    ZStack(alignment: .topLeading) {
                        Image(uiImage: refImage).resizable().scaledToFit()
                            .frame(width: geo.size.width, height: geo.size.height)
                        if let ov = selectionOverlay {
                            Image(decorative: ov, scale: 1).resizable().interpolation(.none).scaledToFit()
                                .frame(width: geo.size.width, height: geo.size.height)
                                .opacity(0.55).allowsHitTesting(false)
                        }
                        if dragRect.width > 1 {
                            Rectangle().fill(Color.yellow.opacity(0.18))
                                .frame(width: dragRect.width, height: dragRect.height)
                                .overlay(Rectangle().stroke(Color.yellow, lineWidth: 3))
                                .position(x: dragRect.midX, y: dragRect.midY)
                        }
                        if isSelecting { ProgressView().tint(.white).position(x: geo.size.width / 2, y: geo.size.height / 2) }
                    }
                    .contentShape(Rectangle())
                    .gesture(
                        DragGesture(minimumDistance: 0)
                            .onChanged { v in
                                if dist(v.startLocation, v.location) > 8 {
                                    let a = clamp(v.startLocation, fit), b = clamp(v.location, fit)
                                    dragRect = CGRect(x: min(a.x, b.x), y: min(a.y, b.y),
                                                      width: abs(a.x - b.x), height: abs(a.y - b.y))
                                }
                            }
                            .onEnded { v in
                                let drag = dist(v.startLocation, v.location)
                                if drag <= 8 {                              // tap → segment the object
                                    let p = clamp(v.startLocation, fit)
                                    dragRect = .zero
                                    selectByTap(CGPoint(x: (p.x - fit.minX) / fit.width,
                                                        y: (p.y - fit.minY) / fit.height))
                                } else if dragRect.width > 8, dragRect.height > 8 {   // drag → box
                                    let bn = CGRect(x: (dragRect.minX - fit.minX) / fit.width,
                                                    y: (dragRect.minY - fit.minY) / fit.height,
                                                    width: dragRect.width / fit.width,
                                                    height: dragRect.height / fit.height)
                                    dragRect = .zero
                                    selectByBox(bn)
                                }
                            }
                    )
                }
            } else {
                Spacer()
                VStack(spacing: 16) {
                    Image(systemName: "hand.tap").font(.system(size: 48)).foregroundStyle(.secondary)
                    Text("Capture or pick a reference photo,\nthen tap an object to detect it")
                        .multilineTextAlignment(.center).font(.subheadline).foregroundStyle(.secondary)
                    HStack(spacing: 12) {
                        if cameraAvailable {
                            Button { showCamera = true } label: {
                                Label("Take Photo", systemImage: "camera").font(.subheadline.bold())
                                    .foregroundColor(.white).padding(.horizontal, 16).padding(.vertical, 10)
                                    .background(Color.blue, in: Capsule())
                            }
                        }
                        PhotosPicker(selection: $selectedItem, matching: .images) {
                            Label("Choose Photo", systemImage: "photo").font(.subheadline.bold())
                                .foregroundColor(.white).padding(.horizontal, 16).padding(.vertical, 10)
                                .background(.ultraThinMaterial, in: Capsule())
                        }
                    }
                }
                Spacer()
            }

            VStack(spacing: 8) {
                Text(refImage == nil ? "Detect any object by example — no text needed"
                     : isSelecting ? "Segmenting…"
                     : (activeMask80 == nil ? "Tap an object — or drag a box around it" : "Selected — tap Detect to find it live"))
                    .font(.caption).foregroundColor(.white.opacity(0.85))
                HStack(spacing: 12) {
                    if cameraAvailable {
                        Button { showCamera = true } label: {
                            Label("Camera", systemImage: "camera").font(.subheadline.bold())
                                .foregroundColor(.white).padding(.horizontal, 12).padding(.vertical, 8)
                                .background(.ultraThinMaterial, in: Capsule())
                        }
                    }
                    PhotosPicker(selection: $selectedItem, matching: .images) {
                        Label("Photo", systemImage: "photo").font(.subheadline.bold())
                            .foregroundColor(.white).padding(.horizontal, 12).padding(.vertical, 8)
                            .background(.ultraThinMaterial, in: Capsule())
                    }
                    Spacer()
                    Button { startDetection() } label: {
                        Label("Detect", systemImage: "viewfinder").font(.subheadline.bold())
                            .foregroundColor(.white).padding(.horizontal, 16).padding(.vertical, 8)
                            .background(activeMask80 == nil ? Color.gray : Color.blue, in: Capsule())
                    }.disabled(activeMask80 == nil)
                }
            }
            .padding(.horizontal, 16).padding(.vertical, 12)
            .background(.ultraThinMaterial)
        }
    }

    private func selectByTap(_ tapNorm: CGPoint) {
        guard let img = refImage, !isSelecting else { return }
        isSelecting = true
        Task {
            let m = detector.maskFromTap(referenceImage: img, tapNorm: tapNorm)
            let ov = m.flatMap { detector.renderMask80($0, referenceImage: img) }
            await MainActor.run {
                isSelecting = false
                if let m { activeMask80 = m; selectionOverlay = ov }   // keep prior selection on a miss
            }
        }
    }

    private func selectByBox(_ boxNorm: CGRect) {
        guard let img = refImage, let m = detector.boxMask80(referenceImage: img, box: boxNorm) else { return }
        activeMask80 = m
        selectionOverlay = detector.renderMask80(m, referenceImage: img)
    }

    private func startDetection() {
        guard activeMask80 != nil else { return }
        applyVisualQuery()
        queryActive = true
    }

    private func applyVisualQuery() {
        guard let refImage, let mask80 = activeMask80 else { return }
        detector.encodeVisualMask(referenceImage: refImage, mask80: mask80)
    }

    private func clearSelection() {
        dragRect = .zero; activeMask80 = nil; selectionOverlay = nil
    }

    private func loadRef() {
        guard let selectedItem else { return }
        clearSelection()
        Task {
            if let data = try? await selectedItem.loadTransferable(type: Data.self),
               let ui = UIImage(data: data) {
                let up = uprightImage(ui)
                await MainActor.run { refImage = up }
            }
        }
    }

    private func dist(_ a: CGPoint, _ b: CGPoint) -> CGFloat { hypot(a.x - b.x, a.y - b.y) }

    /// Bake any EXIF/camera orientation (camera captures are `.right` = 90° rotated) into the
    /// pixels and cap the size, so the displayed image, the drawn box, and the encoded image all
    /// share one upright coordinate system.
    private func uprightImage(_ image: UIImage, maxDim: CGFloat = 1280) -> UIImage {
        let size = image.size
        let s = min(1, maxDim / max(size.width, size.height, 1))
        let target = CGSize(width: max(1, size.width * s), height: max(1, size.height * s))
        let format = UIGraphicsImageRendererFormat.default()
        format.scale = 1
        return UIGraphicsImageRenderer(size: target, format: format).image { _ in
            image.draw(in: CGRect(origin: .zero, size: target))
        }
    }

    private func fittedRect(_ imageSize: CGSize, _ container: CGSize) -> CGRect {
        guard imageSize.width > 0, imageSize.height > 0 else { return .zero }
        let s = min(container.width / imageSize.width, container.height / imageSize.height)
        let w = imageSize.width * s, h = imageSize.height * s
        return CGRect(x: (container.width - w) / 2, y: (container.height - h) / 2, width: w, height: h)
    }

    private func clamp(_ p: CGPoint, _ r: CGRect) -> CGPoint {
        CGPoint(x: min(max(p.x, r.minX), r.maxX), y: min(max(p.y, r.minY), r.maxY))
    }
}

// MARK: - Camera Capture (system camera, for a visual-prompt reference photo)

struct CameraPicker: UIViewControllerRepresentable {
    var onImage: (UIImage) -> Void
    @Environment(\.dismiss) private var dismiss

    func makeUIViewController(context: Context) -> UIImagePickerController {
        let picker = UIImagePickerController()
        picker.sourceType = .camera
        picker.delegate = context.coordinator
        return picker
    }
    func updateUIViewController(_ vc: UIImagePickerController, context: Context) {}
    func makeCoordinator() -> Coordinator { Coordinator(self) }

    final class Coordinator: NSObject, UIImagePickerControllerDelegate, UINavigationControllerDelegate {
        let parent: CameraPicker
        init(_ parent: CameraPicker) { self.parent = parent }
        func imagePickerController(_ picker: UIImagePickerController,
                                   didFinishPickingMediaWithInfo info: [UIImagePickerController.InfoKey: Any]) {
            if let img = (info[.editedImage] as? UIImage) ?? (info[.originalImage] as? UIImage) {
                parent.onImage(img)
            }
            parent.dismiss()
        }
        func imagePickerControllerDidCancel(_ picker: UIImagePickerController) { parent.dismiss() }
    }
}

// MARK: - Bounding Box View (CALayer pool for camera)

class BoundingBoxView {
    let shapeLayer = CAShapeLayer()
    let fillLayer = CAShapeLayer()
    let textLayer = CATextLayer()

    init() {
        fillLayer.isHidden = true
        shapeLayer.fillColor = nil
        shapeLayer.lineWidth = 3
        shapeLayer.lineCap = .round
        shapeLayer.lineJoin = .round
        shapeLayer.isHidden = true
        textLayer.fontSize = 11
        textLayer.font = UIFont.systemFont(ofSize: 11, weight: .semibold)
        textLayer.foregroundColor = UIColor.white.cgColor
        textLayer.contentsScale = UIScreen.main.scale
        textLayer.isHidden = true
        textLayer.cornerRadius = 8
        textLayer.masksToBounds = true
        textLayer.alignmentMode = .center
    }

    func addToLayer(_ parent: CALayer) {
        parent.addSublayer(fillLayer)
        parent.addSublayer(shapeLayer)
        parent.addSublayer(textLayer)
    }

    func show(frame: CGRect, label: String, color: UIColor, alpha: CGFloat) {
        CATransaction.begin()
        CATransaction.setDisableActions(true)
        let path = UIBezierPath(roundedRect: frame, cornerRadius: 10).cgPath
        shapeLayer.path = path
        shapeLayer.strokeColor = color.withAlphaComponent(alpha).cgColor
        shapeLayer.isHidden = false
        fillLayer.path = path
        fillLayer.fillColor = color.withAlphaComponent(0.08).cgColor
        fillLayer.isHidden = false
        textLayer.string = "  \(label)  "
        textLayer.backgroundColor = color.withAlphaComponent(0.95).cgColor
        let tw = CGFloat(label.count) * 7 + 20
        let ty = frame.minY > 28 ? frame.minY - 24 : frame.maxY + 4
        textLayer.frame = CGRect(x: frame.minX, y: ty,
                                 width: min(tw, max(frame.width + 24, 64)), height: 20)
        textLayer.isHidden = false
        CATransaction.commit()
    }

    func hide() {
        CATransaction.begin()
        CATransaction.setDisableActions(true)
        shapeLayer.isHidden = true
        fillLayer.isHidden = true
        textLayer.isHidden = true
        CATransaction.commit()
    }
}

// MARK: - Mask Data Container

struct MaskData {
    let coeffs: [Float]   // packed [32, numAnchors]
    let protos: [Float]   // packed [32, 160*160]
    let numAnchors: Int
}

// MARK: - Detection Result with Masks

struct DetectionResult {
    let detections: [Detection]
    let maskData: MaskData?
    /// Combined instance-mask overlay at proto resolution, de-letterboxed to the
    /// original-image aspect (built for the live camera/video paths). nil otherwise.
    var maskImage: CGImage? = nil
}

// MARK: - Text Grounding Detector

class TextGroundingDetector: ObservableObject {
    @Published var isModelLoaded = false
    @Published var lastMaxScore: Float = 0   // best pre-threshold match (for the Visual tab readout)

    let colors: [UIColor] = [
        .systemRed, .systemGreen, .systemBlue, .systemOrange,
        .systemPurple, .systemYellow, .systemPink, .systemCyan,
    ]

    private var visualModel: MLModel?            // yoloe_detector: image -> boxes, region_embeddings, masks
    private var textEncoder: MLModel?            // Apple mobileclip_blt_text: text[1,77] -> final_emb_1[1,512]
    private var reprtaModel: MLModel?            // YOLOE reprta: raw_tpe[1,80,512] -> tpe[1,80,512]
    private var visualEncoder: MLModel?          // YOLOE SAVPE: image + mask[1,1,80,80] -> vpe[1,1,512]
    private var tokenizer: CLIPTokenizer?

    private let embedDim = 512
    private let augDim = 513                      // embed + 1 (bias channel of the contrastive head)
    private let reprtaSlots = 80                  // reprta input is a fixed [1,80,512] buffer
    private let numAnchors = 8400
    private let inputSize = 640
    private let maskSize = 80                     // SAVPE prompt-mask resolution (640 / 8)
    var confidenceThreshold: Float = 0.15
    var maxDetections: Int = 50          // cap on returned detections (1...100), user-settable
    private let nmsThreshold: Float = 0.5
    private let ciContext = CIContext(options: [.useSoftwareRenderer: false])

    private var imageArray: MLMultiArray?
    private var cachedQueryString = ""
    private(set) var cachedQueries: [String] = []
    /// Cached query embeddings, row-major [N, 513] = [normalize(query), 1.0]. The query
    /// can be text (MobileCLIP -> reprta) or visual (SAVPE) — both live in the same space.
    /// Per-frame similarity is logit = query' · region', score = sigmoid(logit),
    /// reproducing YOLOE's BNContrastiveHead exactly.
    private var cachedQueryPrime: [Float] = []

    /// Last mask data from the most recent detection (for threshold re-rendering)
    var lastMaskData: MaskData?

    init() {
        loadModels()
    }

    private func loadModels() {
        do {
            guard let d = Bundle.main.url(forResource: "yoloe_detector", withExtension: "mlmodelc"),
                  let e = Bundle.main.url(forResource: "mobileclip_blt_text", withExtension: "mlmodelc"),
                  let r = Bundle.main.url(forResource: "reprta", withExtension: "mlmodelc"),
                  let v = Bundle.main.url(forResource: "clip_vocab", withExtension: "json") else {
                print("[YOLOE] Missing model files")
                return
            }
            let config = MLModelConfiguration()
            config.computeUnits = .all
            visualModel = try MLModel(contentsOf: d, configuration: config)
            textEncoder = try MLModel(contentsOf: e, configuration: config)
            reprtaModel = try MLModel(contentsOf: r, configuration: config)
            tokenizer = try CLIPTokenizer(vocabularyURL: v)
            // Optional: visual prompt encoder (SAVPE). Detection still works without it.
            if let vp = Bundle.main.url(forResource: "visual_prompt_encoder", withExtension: "mlmodelc") {
                visualEncoder = try MLModel(contentsOf: vp, configuration: config)
            }
            DispatchQueue.main.async { self.isModelLoaded = true }
        } catch {
            print("[YOLOE] Model load failed: \(error)")
        }
    }

    // MARK: - Text Encoding

    func updateQueries(_ queryString: String) {
        guard queryString != cachedQueryString else { return }
        cachedQueryString = queryString

        let queries = queryString.split(separator: ",").map {
            $0.trimmingCharacters(in: .whitespaces)
        }.filter { !$0.isEmpty }

        guard !queries.isEmpty, let textEncoder, let reprtaModel, let tokenizer else {
            cachedQueries = []; cachedQueryPrime = []; return
        }
        let n = min(queries.count, reprtaSlots)
        cachedQueries = Array(queries.prefix(n))

        do {
            // 1) Encode each query with Apple's MobileCLIP, L2-normalize, and pack
            //    into the fixed [1, 80, 512] reprta input buffer.
            let rawTpe = try MLMultiArray(shape: [1, reprtaSlots as NSNumber, embedDim as NSNumber], dataType: .float32)
            let rawPtr = rawTpe.dataPointer.bindMemory(to: Float32.self, capacity: reprtaSlots * embedDim)
            memset(rawPtr, 0, reprtaSlots * embedDim * 4)

            for (i, query) in cachedQueries.enumerated() {
                let tokens = tokenizer.tokenize(query)
                let tokenArray = try MLMultiArray(shape: [1, tokenizer.contextLength as NSNumber], dataType: .int32)
                let tokenPtr = tokenArray.dataPointer.bindMemory(to: Int32.self, capacity: tokenizer.contextLength)
                for j in 0..<tokenizer.contextLength { tokenPtr[j] = Int32(tokens[j]) }

                let input = try MLDictionaryFeatureProvider(dictionary: ["text": MLFeatureValue(multiArray: tokenArray)])
                let output = try textEncoder.prediction(from: input)
                guard let embMA = output.featureValue(for: "final_emb_1")?.multiArrayValue else { continue }

                let emb = readFloat(embMA)
                var norm: Float = 0
                vDSP_svesq(emb, 1, &norm, vDSP_Length(Int(embedDim)))
                norm = sqrt(norm)
                if norm > 1e-8 {
                    for j in 0..<embedDim { rawPtr[i * embedDim + j] = emb[j] / norm }
                }
            }

            // 2) RepRTA residual MLP (raw_tpe -> tpe). Normalization is done here.
            let reprtaInput = try MLDictionaryFeatureProvider(dictionary: ["raw_tpe": MLFeatureValue(multiArray: rawTpe)])
            let reprtaOutput = try reprtaModel.prediction(from: reprtaInput)
            guard let tpeMA = reprtaOutput.featureValue(for: "tpe")?.multiArrayValue else {
                cachedQueryPrime = []; return
            }
            let tpe = readFloat(tpeMA)  // [1, 80, 512]

            // 3) Build text' = [normalize(tpe), 1.0] -> row-major [N, 513].
            var textPrime = [Float](repeating: 0, count: n * augDim)
            for i in 0..<n {
                let off = i * embedDim
                var norm: Float = 0
                tpe.withUnsafeBufferPointer { vDSP_svesq($0.baseAddress! + off, 1, &norm, vDSP_Length(Int(embedDim))) }
                let inv: Float = norm > 1e-8 ? 1.0 / sqrt(norm) : 0
                for c in 0..<embedDim { textPrime[i * augDim + c] = tpe[off + c] * inv }
                textPrime[i * augDim + embedDim] = 1.0  // bias channel multiplier
            }
            cachedQueryPrime = textPrime
        } catch {
            cachedQueries = []; cachedQueryPrime = []
        }
    }

    // MARK: - Visual query (detect "the same object" as a reference region)

    /// Encode an 80x80 reference mask (640-letterbox space) into the cached visual prompt
    /// embedding via SAVPE — a drop-in for the text query. The mask can be a box rasterization
    /// or an object-shaped mask (tap-to-segment); SAVPE just pools features over the masked
    /// region, so a tighter mask gives a cleaner embedding.
    func encodeVisualMask(referenceImage: UIImage, mask80: [Float], label: String = "Object") {
        guard let visualEncoder, mask80.count == maskSize * maskSize,
              let cg = normalizedCGImage(referenceImage) else {
            cachedQueries = []; cachedQueryPrime = []; return
        }
        cachedQueryString = "\u{1F5BC} " + label   // marker so a later text update still applies
        do {
            let (tensor, _, _, _, _, _) = try preprocessImage(cg, fresh: true)
            let mask = try MLMultiArray(shape: [1, 1, maskSize as NSNumber, maskSize as NSNumber], dataType: .float32)
            let mp = mask.dataPointer.bindMemory(to: Float32.self, capacity: maskSize * maskSize)
            var any = false
            for i in 0..<(maskSize * maskSize) { mp[i] = mask80[i]; if mask80[i] > 0 { any = true } }
            guard any else { cachedQueries = []; cachedQueryPrime = []; return }

            let input = try MLDictionaryFeatureProvider(dictionary: [
                "image": MLFeatureValue(multiArray: tensor),
                "mask": MLFeatureValue(multiArray: mask),
            ])
            let out = try visualEncoder.prediction(from: input)
            guard let vpeMA = out.featureValue(for: "vpe")?.multiArrayValue else {
                cachedQueries = []; cachedQueryPrime = []; return
            }
            let vpe = readFloat(vpeMA)   // [1,1,512], already L2-normalized by SAVPE
            var q = [Float](repeating: 0, count: augDim)
            for c in 0..<embedDim { q[c] = vpe[c] }
            q[embedDim] = 1.0
            cachedQueries = [label]
            cachedQueryPrime = q
        } catch {
            cachedQueries = []; cachedQueryPrime = []
        }
    }

    /// Rasterize a box (normalized reference coords) into an 80x80 mask (640-letterbox space).
    func boxMask80(referenceImage: UIImage, box: CGRect) -> [Float]? {
        guard let cg = normalizedCGImage(referenceImage) else { return nil }
        let imgW = cg.width, imgH = cg.height
        let scale = Float(inputSize) / Float(max(imgW, imgH))
        let padX = (Float(inputSize) - Float(imgW) * scale) / 2
        let padY = (Float(inputSize) - Float(imgH) * scale) / 2
        let s8 = Float(maskSize) / Float(inputSize)
        func toMask(_ nx: CGFloat, _ ny: CGFloat) -> (Int, Int) {
            (Int(((Float(nx) * Float(imgW) * scale + padX) * s8).rounded()),
             Int(((Float(ny) * Float(imgH) * scale + padY) * s8).rounded()))
        }
        let (mx0, my0) = toMask(box.minX, box.minY), (mx1, my1) = toMask(box.maxX, box.maxY)
        let x0 = max(0, min(maskSize - 1, mx0)), x1 = max(0, min(maskSize, mx1))
        let y0 = max(0, min(maskSize - 1, my0)), y1 = max(0, min(maskSize, my1))
        guard x1 > x0, y1 > y0 else { return nil }
        var m = [Float](repeating: 0, count: maskSize * maskSize)
        for y in y0..<y1 { for x in x0..<x1 { m[y * maskSize + x] = 1 } }
        return m
    }

    /// Tap → object mask via YOLOE's own segmentation. Runs the detector on the reference,
    /// finds the instance whose mask is strongest at the tapped point, and returns its 80x80
    /// mask (640-letterbox space). nil if nothing convincing is under the tap. `tapNorm` is
    /// normalized reference-image coords [0,1].
    func maskFromTap(referenceImage: UIImage, tapNorm: CGPoint) -> [Float]? {
        guard let visualModel, let cg = normalizedCGImage(referenceImage) else { return nil }
        do {
            let (tensor, imgW, imgH, padX, padY, scale) = try preprocessImage(cg, fresh: true)
            let out = try visualModel.prediction(from: try MLDictionaryFeatureProvider(
                dictionary: ["image": MLFeatureValue(multiArray: tensor)]))
            guard let boxesMA = out.featureValue(for: "boxes")?.multiArrayValue,
                  let coeffsMA = out.featureValue(for: "mask_coeffs")?.multiArrayValue,
                  let protosMA = out.featureValue(for: "mask_protos")?.multiArrayValue else { return nil }
            let boxes = readMatrix2D(boxesMA)     // [4, 8400]
            let coeffs = readMatrix2D(coeffsMA)   // [32, 8400]
            let protos = readProtos(protosMA)     // [32, 25600]
            let nmc = 32, proto = 160, phw = proto * proto

            let tx = Float(tapNorm.x) * Float(imgW) * scale + padX
            let ty = Float(tapNorm.y) * Float(imgH) * scale + padY
            let pIdx = max(0, min(proto - 1, Int(ty / 4))) * proto + max(0, min(proto - 1, Int(tx / 4)))

            // Anchors whose decoded box contains the tap → pick the strongest mask logit at the tap.
            var bestA = -1, bestM: Float = 0
            for a in 0..<numAnchors {
                let cx = boxes[a], cy = boxes[numAnchors + a]
                let bw = boxes[2 * numAnchors + a], bh = boxes[3 * numAnchors + a]
                if tx < cx - bw / 2 || tx > cx + bw / 2 || ty < cy - bh / 2 || ty > cy + bh / 2 { continue }
                var m: Float = 0
                for k in 0..<nmc { m += coeffs[k * numAnchors + a] * protos[k * phw + pIdx] }
                if m > bestM { bestM = m; bestA = a }
            }
            guard bestA >= 0 else { return nil }

            var m160 = [Float](repeating: 0, count: phw)
            for k in 0..<nmc {
                let c = coeffs[k * numAnchors + bestA], base = k * phw
                for p in 0..<phw { m160[p] += c * protos[base + p] }
            }
            var on = 0
            for p in 0..<phw where m160[p] > 0 { on += 1 }
            guard on > 4, Float(on) < 0.7 * Float(phw) else { return nil }   // reject background-ish blobs

            var m80 = [Float](repeating: 0, count: maskSize * maskSize)
            for y in 0..<maskSize { for x in 0..<maskSize {
                m80[y * maskSize + x] = m160[(2 * y) * proto + (2 * x)] > 0 ? 1 : 0
            } }
            return m80
        } catch { return nil }
    }

    /// Render an 80x80 reference mask as a translucent overlay CGImage de-letterboxed to the
    /// original-image aspect (for selection feedback on the reference photo).
    func renderMask80(_ mask80: [Float], referenceImage: UIImage, color: UIColor = .systemGreen) -> CGImage? {
        guard mask80.count == maskSize * maskSize, let cg = normalizedCGImage(referenceImage) else { return nil }
        let imgW = cg.width, imgH = cg.height
        let scale = Float(inputSize) / Float(max(imgW, imgH))
        let s8 = Float(maskSize) / Float(inputSize)
        let padX = Int(((Float(inputSize) - Float(imgW) * scale) / 2 * s8).rounded())
        let padY = Int(((Float(inputSize) - Float(imgH) * scale) / 2 * s8).rounded())
        var cr: CGFloat = 0, cg2: CGFloat = 0, cb: CGFloat = 0, ca: CGFloat = 0
        color.getRed(&cr, green: &cg2, blue: &cb, alpha: &ca)
        var px = [UInt8](repeating: 0, count: maskSize * maskSize * 4)
        for i in 0..<(maskSize * maskSize) where mask80[i] > 0 {
            let o = i * 4
            px[o] = UInt8(cr * 255); px[o + 1] = UInt8(cg2 * 255); px[o + 2] = UInt8(cb * 255); px[o + 3] = 255
        }
        let cs = CGColorSpaceCreateDeviceRGB()
        let info = CGBitmapInfo(rawValue: CGImageAlphaInfo.premultipliedLast.rawValue)
        guard let provider = CGDataProvider(data: Data(px) as CFData),
              let full = CGImage(width: maskSize, height: maskSize, bitsPerComponent: 8, bitsPerPixel: 32,
                                 bytesPerRow: maskSize * 4, space: cs, bitmapInfo: info, provider: provider,
                                 decode: nil, shouldInterpolate: false, intent: .defaultIntent) else { return nil }
        let cw = Int((Float(imgW) * scale * s8).rounded()), ch = Int((Float(imgH) * scale * s8).rounded())
        let crop = CGRect(x: padX, y: padY, width: max(1, min(cw, maskSize - padX)), height: max(1, min(ch, maskSize - padY)))
        return full.cropping(to: crop) ?? full
    }

    // MARK: - Sync Detection (camera / video -- boxes + combined mask overlay)

    func detectSync(pixelBuffer: CVPixelBuffer) -> DetectionResult {
        guard let cgImage = cgImageFromPixelBuffer(pixelBuffer) else {
            return DetectionResult(detections: [], maskData: nil)
        }
        return runDetection(cgImage: cgImage, needMasks: true)
    }

    func detectSync(image: UIImage) -> DetectionResult {
        guard let cgImage = normalizedCGImage(image) else {
            return DetectionResult(detections: [], maskData: nil)
        }
        return runDetection(cgImage: cgImage, needMasks: true)
    }

    // MARK: - Detection with Masks (for photo mode)

    func detectSyncWithMasks(image: UIImage) -> DetectionResult {
        guard let cgImage = normalizedCGImage(image) else { return DetectionResult(detections: [], maskData: nil) }
        let result = runDetection(cgImage: cgImage, needMasks: true)
        lastMaskData = result.maskData
        return result
    }

    private func cgImageFromPixelBuffer(_ pb: CVPixelBuffer) -> CGImage? {
        let ci = CIImage(cvPixelBuffer: pb)
        return ciContext.createCGImage(ci, from: ci.extent)
    }

    // MARK: - Core Detection

    private func runDetection(cgImage: CGImage, needMasks: Bool) -> DetectionResult {
        // Snapshot the text cache so a concurrent updateQueries() can't tear it.
        let textPrime = cachedQueryPrime
        let queries = cachedQueries
        guard let visualModel, !textPrime.isEmpty, queries.count == textPrime.count / augDim else {
            return DetectionResult(detections: [], maskData: nil)
        }
        let n = queries.count

        do {
            let (tensor, imgW, imgH, padX, padY, scale) = try preprocessImage(cgImage)
            let input = try MLDictionaryFeatureProvider(dictionary: ["image": MLFeatureValue(multiArray: tensor)])
            let output = try visualModel.prediction(from: input)

            guard let boxesMA = output.featureValue(for: "boxes")?.multiArrayValue,
                  let regionMA = output.featureValue(for: "region_embeddings")?.multiArrayValue else {
                return DetectionResult(detections: [], maskData: nil)
            }

            // boxes [4, 8400] (xywh @640) and region' [513, 8400], packed FP32.
            let boxes = readMatrix2D(boxesMA)
            let region = readMatrix2D(regionMA)

            // logits [n, 8400] = textPrime [n, 513] x region' [513, 8400].
            // This reproduces YOLOE's BNContrastiveHead exactly; score = sigmoid(logit).
            var logits = [Float](repeating: 0, count: n * numAnchors)
            textPrime.withUnsafeBufferPointer { aP in
                region.withUnsafeBufferPointer { bP in
                    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                                Int32(n), Int32(numAnchors), Int32(augDim),
                                1.0, aP.baseAddress, Int32(augDim),
                                bP.baseAddress, Int32(numAnchors),
                                0.0, &logits, Int32(numAnchors))
                }
            }

            // Best match (pre-threshold) — surfaced for the Visual tab so a weak match is visible.
            var maxLogit: Float = -.greatestFiniteMagnitude
            vDSP_maxv(logits, 1, &maxLogit, vDSP_Length(logits.count))
            let best = 1.0 / (1.0 + exp(-maxLogit))
            DispatchQueue.main.async { if abs(self.lastMaxScore - best) > 0.005 { self.lastMaxScore = best } }

            // Threshold on the logit so sigmoid is only evaluated for survivors.
            let t = confidenceThreshold
            let logitThresh: Float = (t > 0 && t < 1) ? log(t / (1 - t)) : -.greatestFiniteMagnitude
            let invW = 1.0 / (Float(imgW) * scale)
            let invH = 1.0 / (Float(imgH) * scale)

            var allDets: [(CGRect, Float, Int, Int)] = []  // rect, score, classIdx, anchorIdx
            for k in 0..<n {
                let off = k * numAnchors
                for a in 0..<numAnchors {
                    let logit = logits[off + a]
                    guard logit >= logitThresh else { continue }
                    let score = 1.0 / (1.0 + exp(-logit))

                    let cx = boxes[a], cy = boxes[numAnchors + a]
                    let bw = boxes[2 * numAnchors + a], bh = boxes[3 * numAnchors + a]
                    let nx = (cx - bw / 2 - padX) * invW
                    let ny = (cy - bh / 2 - padY) * invH
                    let rect = CGRect(
                        x: CGFloat(max(0, min(1, nx))),
                        y: CGFloat(max(0, min(1, ny))),
                        width: CGFloat(max(0, min(1, bw * invW))),
                        height: CGFloat(max(0, min(1, bh * invH)))
                    )
                    allDets.append((rect, score, k, a))
                }
            }

            // Per-class NMS.
            allDets.sort { $0.1 > $1.1 }
            var kept: [Int] = []
            for i in allDets.indices {
                var suppress = false
                for ki in kept where allDets[i].2 == allDets[ki].2 {
                    if iou(allDets[i].0, allDets[ki].0) > nmsThreshold { suppress = true; break }
                }
                if !suppress { kept.append(i) }
            }

            let detections = kept.prefix(max(1, min(100, maxDetections))).map { i in
                Detection(label: queries[allDets[i].2],
                          confidence: allDets[i].1,
                          classIndex: allDets[i].2,
                          normRect: allDets[i].0,
                          anchorIndex: allDets[i].3)
            }

            var maskData: MaskData? = nil
            var maskImage: CGImage? = nil
            if needMasks,
               let coeffsMA = output.featureValue(for: "mask_coeffs")?.multiArrayValue,
               let protosMA = output.featureValue(for: "mask_protos")?.multiArrayValue {
                let md = MaskData(coeffs: readMatrix2D(coeffsMA),
                                  protos: readProtos(protosMA),
                                  numAnchors: numAnchors)
                maskData = md
                maskImage = buildCombinedMask(detections, md,
                                              padX: padX, padY: padY, scale: scale,
                                              imgW: imgW, imgH: imgH)
            }
            return DetectionResult(detections: detections, maskData: maskData, maskImage: maskImage)
        } catch {
            return DetectionResult(detections: [], maskData: nil)
        }
    }

    // MARK: - Preprocessing

    private func preprocessImage(_ cgImage: CGImage, fresh: Bool = false) throws
        -> (MLMultiArray, Int, Int, Float, Float, Float)
    {
        let imgW = cgImage.width, imgH = cgImage.height
        let scale = Float(inputSize) / Float(max(imgW, imgH))
        let scaledW = Int(Float(imgW) * scale)
        let scaledH = Int(Float(imgH) * scale)
        let padX = (inputSize - scaledW) / 2
        let padY = (inputSize - scaledH) / 2

        // Use UIGraphicsImageRenderer (UIKit y-down coordinates) to avoid
        // CGContext's y-up coordinate system flipping the image.
        let renderer = UIGraphicsImageRenderer(size: CGSize(width: inputSize, height: inputSize))
        let uiImage = renderer.image { ctx in
            UIColor.gray.setFill()
            ctx.fill(CGRect(x: 0, y: 0, width: inputSize, height: inputSize))
            UIImage(cgImage: cgImage).draw(in: CGRect(x: padX, y: padY, width: scaledW, height: scaledH))
        }
        guard let rendered = uiImage.cgImage else { throw NSError(domain: "Preprocess", code: 1) }
        guard let ctx = CGContext(
            data: nil, width: inputSize, height: inputSize,
            bitsPerComponent: 8, bytesPerRow: inputSize * 4,
            space: CGColorSpaceCreateDeviceRGB(),
            bitmapInfo: CGImageAlphaInfo.noneSkipLast.rawValue
        ) else { throw NSError(domain: "Preprocess", code: 2) }
        ctx.draw(rendered, in: CGRect(x: 0, y: 0, width: inputSize, height: inputSize))
        guard let pixels = ctx.data else { throw NSError(domain: "Preprocess", code: 3) }

        // `fresh` allocates a private buffer (visual-prompt path) so it can't race the
        // camera's shared imageArray on the inference queue.
        let arr: MLMultiArray
        if fresh {
            arr = try MLMultiArray(shape: [1, 3, inputSize as NSNumber, inputSize as NSNumber], dataType: .float32)
        } else {
            if imageArray == nil {
                imageArray = try MLMultiArray(
                    shape: [1, 3, inputSize as NSNumber, inputSize as NSNumber], dataType: .float32)
            }
            arr = imageArray!
        }
        let dst = arr.dataPointer.bindMemory(to: Float32.self, capacity: 3 * inputSize * inputSize)
        let src = pixels.bindMemory(to: UInt8.self, capacity: inputSize * inputSize * 4)

        let hw = inputSize * inputSize
        let inv: Float = 1.0 / 255.0
        for i in 0..<hw {
            dst[0 * hw + i] = Float(src[i * 4 + 0]) * inv
            dst[1 * hw + i] = Float(src[i * 4 + 1]) * inv
            dst[2 * hw + i] = Float(src[i * 4 + 2]) * inv
        }

        return (arr, imgW, imgH, Float(padX), Float(padY), scale)
    }

    /// Normalize UIImage orientation so cgImage matches the displayed orientation.
    private func normalizedCGImage(_ image: UIImage) -> CGImage? {
        guard image.imageOrientation != .up else { return image.cgImage }
        UIGraphicsBeginImageContextWithOptions(image.size, false, image.scale)
        image.draw(in: CGRect(origin: .zero, size: image.size))
        let normalized = UIGraphicsGetImageFromCurrentImageContext()
        UIGraphicsEndImageContext()
        return normalized?.cgImage
    }

    // MARK: - Helpers

    private func iou(_ a: CGRect, _ b: CGRect) -> Float {
        let interX = max(0, min(a.maxX, b.maxX) - max(a.minX, b.minX))
        let interY = max(0, min(a.maxY, b.maxY) - max(a.minY, b.minY))
        let inter = Float(interX * interY)
        let union = Float(a.width * a.height) + Float(b.width * b.height) - inter
        return union > 0 ? inter / union : 0
    }

    private func readFloat(_ array: MLMultiArray) -> [Float] {
        let count = array.count
        var result = [Float](repeating: 0, count: count)
        if array.dataType == .float16 {
            let ptr = array.dataPointer.assumingMemoryBound(to: Float16.self)
            for i in 0..<count { result[i] = Float(ptr[i]) }
        } else {
            let ptr = array.dataPointer.assumingMemoryBound(to: Float32.self)
            for i in 0..<count { result[i] = ptr[i] }
        }
        return result
    }

    /// Read an MLMultiArray whose trailing two dims are [rows, cols] into a packed
    /// row-major FP32 buffer, de-padding ANE row alignment. FP16 outputs are bulk
    /// converted with vImage -- never element-by-element (see conversion notes:
    /// boxing 4M FP16 values per frame costs ~170 ms).
    private func readMatrix2D(_ a: MLMultiArray) -> [Float] {
        let shape = a.shape.map { $0.intValue }
        let strides = a.strides.map { $0.intValue }
        let rows = shape[shape.count - 2]
        let cols = shape[shape.count - 1]
        let rowStride = strides[strides.count - 2]
        let colStride = strides[strides.count - 1]
        var out = [Float](repeating: 0, count: rows * cols)

        if colStride == 1 {
            out.withUnsafeMutableBufferPointer { dst in
                if a.dataType == .float16 {
                    var s = vImage_Buffer(data: a.dataPointer, height: vImagePixelCount(rows),
                                          width: vImagePixelCount(cols), rowBytes: rowStride * 2)
                    var d = vImage_Buffer(data: UnsafeMutableRawPointer(dst.baseAddress!),
                                          height: vImagePixelCount(rows),
                                          width: vImagePixelCount(cols), rowBytes: cols * 4)
                    vImageConvert_Planar16FtoPlanarF(&s, &d, vImage_Flags(0))
                } else {
                    let src = a.dataPointer.assumingMemoryBound(to: Float32.self)
                    for r in 0..<rows { memcpy(dst.baseAddress! + r * cols, src + r * rowStride, cols * 4) }
                }
            }
        } else {  // general strided fallback
            if a.dataType == .float16 {
                let p = a.dataPointer.assumingMemoryBound(to: Float16.self)
                for r in 0..<rows { for c in 0..<cols { out[r * cols + c] = Float(p[r * rowStride + c * colStride]) } }
            } else {
                let p = a.dataPointer.assumingMemoryBound(to: Float32.self)
                for r in 0..<rows { for c in 0..<cols { out[r * cols + c] = p[r * rowStride + c * colStride] } }
            }
        }
        return out
    }

    /// Read mask protos [1, C, H, W] into packed [C, H*W] FP32, handling ANE padding.
    private func readProtos(_ a: MLMultiArray) -> [Float] {
        let shape = a.shape.map { $0.intValue }
        let strides = a.strides.map { $0.intValue }
        let c = shape[1], h = shape[2], w = shape[3]
        let sc = strides[1], sh = strides[2], sw = strides[3]
        var out = [Float](repeating: 0, count: c * h * w)

        if sw == 1 && sh == w {  // contiguous rows; channels may still be padded
            out.withUnsafeMutableBufferPointer { dst in
                if a.dataType == .float16 {
                    var s = vImage_Buffer(data: a.dataPointer, height: vImagePixelCount(c),
                                          width: vImagePixelCount(h * w), rowBytes: sc * 2)
                    var d = vImage_Buffer(data: UnsafeMutableRawPointer(dst.baseAddress!),
                                          height: vImagePixelCount(c),
                                          width: vImagePixelCount(h * w), rowBytes: h * w * 4)
                    vImageConvert_Planar16FtoPlanarF(&s, &d, vImage_Flags(0))
                } else {
                    let src = a.dataPointer.assumingMemoryBound(to: Float32.self)
                    for ch in 0..<c { memcpy(dst.baseAddress! + ch * h * w, src + ch * sc, h * w * 4) }
                }
            }
        } else {  // general strided fallback
            if a.dataType == .float16 {
                let p = a.dataPointer.assumingMemoryBound(to: Float16.self)
                for ch in 0..<c { for y in 0..<h {
                    let base = ch * sc + y * sh, o = (ch * h + y) * w
                    for x in 0..<w { out[o + x] = Float(p[base + x * sw]) }
                } }
            } else {
                let p = a.dataPointer.assumingMemoryBound(to: Float32.self)
                for ch in 0..<c { for y in 0..<h {
                    let base = ch * sc + y * sh, o = (ch * h + y) * w
                    for x in 0..<w { out[o + x] = p[base + x * sw] }
                } }
            }
        }
        return out
    }

    // MARK: - Combined instance-mask overlay (yolo-ios-app fast method)

    /// Build one proto-resolution RGBA overlay for all detections in a single BLAS matmul
    /// (coeffs[N,32] x protos[32,HW]), composite per-bbox lowest-score-first, then crop out
    /// the letterbox so the result matches the original-image aspect. nil if no detections.
    private func buildCombinedMask(_ dets: [Detection], _ md: MaskData,
                                   padX: Float, padY: Float, scale: Float,
                                   imgW: Int, imgH: Int) -> CGImage? {
        let n = dets.count
        guard n > 0 else { return nil }
        let mc = 32, mw = 160, mh = 160, hw = mw * mh

        // Gather coeffs A[n,32] for the kept anchors, then combined[n,hw] = A x protos.
        var a = [Float](repeating: 0, count: n * mc)
        for (i, d) in dets.enumerated() {
            for k in 0..<mc { a[i * mc + k] = md.coeffs[k * md.numAnchors + d.anchorIndex] }
        }
        var comb = [Float](repeating: 0, count: n * hw)
        a.withUnsafeBufferPointer { aP in
            md.protos.withUnsafeBufferPointer { bP in
                cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                            Int32(n), Int32(hw), Int32(mc),
                            1.0, aP.baseAddress, Int32(mc),
                            bP.baseAddress, Int32(hw),
                            0.0, &comb, Int32(hw))
            }
        }

        // Composite into one proto-res RGBA buffer (lowest score first -> highest on top).
        var px = [UInt8](repeating: 0, count: hw * 4)
        let s160 = Float(mw) / Float(inputSize)      // 160 / 640
        let order = dets.indices.sorted { dets[$0].confidence < dets[$1].confidence }
        for i in order {
            let d = dets[i], r = d.normRect
            // original-normalized box -> 640 letterbox -> 160 proto space
            let x0 = (Float(r.minX) * Float(imgW) * scale + padX) * s160
            let y0 = (Float(r.minY) * Float(imgH) * scale + padY) * s160
            let x1 = (Float(r.maxX) * Float(imgW) * scale + padX) * s160
            let y1 = (Float(r.maxY) * Float(imgH) * scale + padY) * s160
            let bx0 = max(0, min(mw - 1, Int(x0))), bx1 = max(0, min(mw - 1, Int(x1)))
            let by0 = max(0, min(mh - 1, Int(y0))), by1 = max(0, min(mh - 1, Int(y1)))
            guard bx1 >= bx0, by1 >= by0 else { continue }
            let (cr, cg, cb) = rgbComponents(colors[d.classIndex % colors.count])
            let base = i * hw
            for y in by0...by1 {
                let row = y * mw
                for x in bx0...bx1 where comb[base + row + x] > 0 {  // logit>0 == sigmoid>0.5
                    let o = (row + x) * 4
                    px[o] = cr; px[o + 1] = cg; px[o + 2] = cb; px[o + 3] = 255
                }
            }
        }

        guard let full = makeRGBA(px, mw, mh) else { return nil }
        // De-letterbox so the overlay matches the original frame (shown with the same gravity).
        let cx = Int((padX * s160).rounded()), cy = Int((padY * s160).rounded())
        let cw = Int((Float(imgW) * scale * s160).rounded())
        let ch = Int((Float(imgH) * scale * s160).rounded())
        let crop = CGRect(x: cx, y: cy, width: max(1, min(cw, mw - cx)), height: max(1, min(ch, mh - cy)))
        return full.cropping(to: crop) ?? full
    }

    private func rgbComponents(_ c: UIColor) -> (UInt8, UInt8, UInt8) {
        var r: CGFloat = 0, g: CGFloat = 0, b: CGFloat = 0, a: CGFloat = 0
        c.getRed(&r, green: &g, blue: &b, alpha: &a)
        return (UInt8(max(0, min(1, r)) * 255), UInt8(max(0, min(1, g)) * 255), UInt8(max(0, min(1, b)) * 255))
    }

    private func makeRGBA(_ px: [UInt8], _ w: Int, _ h: Int) -> CGImage? {
        let cs = CGColorSpaceCreateDeviceRGB()
        let info = CGBitmapInfo(rawValue: CGImageAlphaInfo.premultipliedLast.rawValue)
        guard let provider = CGDataProvider(data: Data(px) as CFData) else { return nil }
        return CGImage(width: w, height: h, bitsPerComponent: 8, bitsPerPixel: 32,
                       bytesPerRow: w * 4, space: cs, bitmapInfo: info, provider: provider,
                       decode: nil, shouldInterpolate: false, intent: .defaultIntent)
    }
}

#Preview { ContentView() }
