import SwiftUI
import PhotosUI
import AVKit
import UniformTypeIdentifiers
import CoreGraphics
import SAMKit

/// FastSAM demo: real-time camera "segment everything", a photo mode (tap to pick one object),
/// and offline video segmentation (burn masks into a new clip). All share the same
/// `FastSamSession` engine from the SAMKit package and the resolution / conf / max controls.
struct ContentView: View {
    private enum Mode: String, CaseIterable { case camera = "Camera", photo = "Photo", video = "Video" }

    @State private var mode: Mode = .camera
    @State private var resolution: Int = 512
    @State private var confidence: Double = 0.5
    @State private var maxObjects: Double = 40
    @StateObject private var camera = CameraController()
    @StateObject private var photo = PhotoModel()
    @StateObject private var video = VideoProcessor()
    @State private var pickerItem: PhotosPickerItem?
    @State private var videoItem: PhotosPickerItem?

    private var modelName: String { "FastSAM_s_\(resolution)" }
    private var currentOptions: FastSamSession.Options {
        FastSamSession.Options(confidenceThreshold: Float(confidence),
                               iouThreshold: 0.9, maxInstances: Int(maxObjects))
    }

    var body: some View {
        VStack(spacing: 0) {
            Picker("Mode", selection: $mode) {
                ForEach(Mode.allCases, id: \.self) { Text($0.rawValue).tag($0) }
            }
            .pickerStyle(.segmented)
            .padding(.horizontal)
            .padding(.top, 8)

            HStack {
                Text("Resolution").font(.caption).foregroundColor(.secondary)
                Picker("Resolution", selection: $resolution) {
                    Text("320").tag(320); Text("512").tag(512); Text("640").tag(640)
                }
                .pickerStyle(.segmented)
                .frame(maxWidth: 240)
                Spacer()
            }
            .padding(.horizontal)
            .padding(.vertical, 6)

            // Tuning sliders (apply to both camera and photo).
            VStack(spacing: 2) {
                HStack {
                    Text(String(format: "Conf %.2f", confidence))
                        .font(.caption2).monospacedDigit().frame(width: 72, alignment: .leading)
                    Slider(value: $confidence, in: 0.2...0.7,
                           onEditingChanged: { editing in if !editing { photo.refreshIfNeeded() } })
                }
                HStack {
                    Text("Max \(Int(maxObjects))")
                        .font(.caption2).monospacedDigit().frame(width: 72, alignment: .leading)
                    Slider(value: $maxObjects, in: 5...100, step: 1,
                           onEditingChanged: { editing in if !editing { photo.refreshIfNeeded() } })
                }
            }
            .padding(.horizontal)
            .onChange(of: confidence) { v in camera.confidence = Float(v); photo.confidence = Float(v) }
            .onChange(of: maxObjects) { v in camera.maxInstances = Int(v); photo.maxInstances = Int(v) }

            switch mode {
            case .camera: cameraView
            case .photo: photoView
            case .video: videoView
            }
        }
        .onAppear {
            camera.confidence = Float(confidence); camera.maxInstances = Int(maxObjects)
            photo.confidence = Float(confidence); photo.maxInstances = Int(maxObjects)
            camera.loadModel(name: modelName)
            photo.loadModel(name: modelName)
            if mode == .camera { camera.start() }
        }
        .onDisappear { camera.stop() }
        .onChange(of: mode) { newMode in
            if newMode == .camera { camera.start() } else { camera.stop() }
        }
        .onChange(of: resolution) { _ in
            camera.loadModel(name: modelName)
            photo.loadModel(name: modelName)
            // re-run photo with new resolution if we already had an image
            photo.refreshIfNeeded()
        }
    }

    // MARK: - Camera (real-time, preview layer + overlay)

    private var cameraView: some View {
        ZStack {
            Color.black
            // System-rendered preview + mask overlay CALayer (both driven outside SwiftUI).
            PreviewView(session: camera.session, controller: camera)
                .ignoresSafeArea(edges: .bottom)

            VStack {
                HStack {
                    Text(String(format: "%.0f FPS", camera.fps))
                        .font(.caption.monospacedDigit()).foregroundColor(.white)
                        .padding(.horizontal, 10).padding(.vertical, 4)
                        .background(Capsule().fill(.black.opacity(0.5)))
                    Spacer()
                    Toggle("Segment", isOn: $camera.enabled).labelsHidden().tint(.green)
                }
                .padding()
                Spacer()
                if !camera.isReady {
                    Text("FastSAM_s_\(resolution).mlpackage not bundled")
                        .font(.caption).foregroundColor(.white)
                        .padding(8)
                        .background(Capsule().fill(.red.opacity(0.7)))
                        .padding(.bottom, 12)
                }
            }
        }
    }

    // MARK: - Photo (tap to pick)

    private var photoView: some View {
        GeometryReader { geo in
            ZStack {
                Color.black
                if let image = photo.image {
                    Image(uiImage: image)
                        .resizable().scaledToFit()
                        .frame(width: geo.size.width, height: geo.size.height)
                        .overlay {
                            if let sel = photo.selected {
                                Image(uiImage: UIImage(cgImage: sel))
                                    .resizable().interpolation(.none).scaledToFit()
                                    .frame(width: geo.size.width, height: geo.size.height)
                                    .opacity(0.6)
                            } else if let overlay = photo.overlay {
                                Image(uiImage: UIImage(cgImage: overlay))
                                    .resizable().interpolation(.none).scaledToFit()
                                    .frame(width: geo.size.width, height: geo.size.height)
                                    .opacity(0.55)
                            }
                        }
                        .contentShape(Rectangle())
                        .onTapGesture { loc in
                            let p = Self.viewToImage(loc, viewSize: geo.size, imageSize: image.size)
                            photo.selectInstance(at: p)
                        }
                } else if !photo.isReady {
                    VStack(spacing: 8) {
                        Image(systemName: "exclamationmark.triangle").font(.largeTitle).foregroundColor(.orange)
                        Text("FastSAM_s_\(resolution).mlpackage not bundled")
                            .font(.subheadline).foregroundColor(.white)
                    }
                } else {
                    VStack(spacing: 12) {
                        Image(systemName: "photo.on.rectangle.angled")
                            .font(.system(size: 50)).foregroundColor(.gray)
                        Text("Pick an image to segment").foregroundColor(.secondary)
                    }
                }

                VStack {
                    Spacer()
                    HStack {
                        PhotosPicker(selection: $pickerItem, matching: .images) {
                            Label(photo.image == nil ? "Pick Image" : "Change Photo", systemImage: "photo")
                                .padding(.vertical, 8)
                        }
                        .buttonStyle(.borderedProminent)
                        if photo.selected != nil {
                            Button("Show all") { photo.clearSelection() }.buttonStyle(.bordered)
                        }
                    }.padding()
                }
                if photo.isProcessing { ProgressView().tint(.white) }
            }
        }
        .onChange(of: pickerItem) { item in
            guard let item else { return }
            Task {
                if let data = try? await item.loadTransferable(type: Data.self),
                   let ui = UIImage(data: data) {
                    photo.setImage(ui.fixedOrientation())
                }
            }
        }
    }

    // MARK: - Video (offline segmentation → burned-in mp4)

    private var videoView: some View {
        VStack(spacing: 14) {
            if video.isProcessing {
                // Live preview of the composited (segmented) frames, like the camera overlay.
                ZStack {
                    Color.black
                    if let preview = video.previewFrame {
                        Image(uiImage: UIImage(cgImage: preview))
                            .resizable().scaledToFit()
                    } else {
                        ProgressView().tint(.white)
                    }
                    VStack {
                        Spacer()
                        ProgressView(value: video.progress)
                            .progressViewStyle(.linear).frame(maxWidth: 260).tint(.white)
                        Text("\(Int(video.progress * 100))%")
                            .font(.headline.monospacedDigit()).foregroundColor(.white)
                            .padding(.bottom, 8)
                    }
                    // DEBUG: exact frame the model receives (no masks). Should look upright,
                    // sharp and natural-coloured — if not, the input is the problem.
                    if let dbg = video.modelInputDebug {
                        VStack {
                            HStack {
                                VStack(alignment: .leading, spacing: 2) {
                                    Text("model input").font(.caption2).foregroundColor(.white)
                                    Image(uiImage: UIImage(cgImage: dbg))
                                        .resizable().scaledToFit()
                                        .frame(width: 110)
                                        .border(Color.white.opacity(0.6))
                                }
                                Spacer()
                            }.padding(8)
                            Spacer()
                        }
                    }
                }
                .frame(maxWidth: .infinity, maxHeight: 420)
                .cornerRadius(10)
            } else if let out = video.outputURL {
                VideoPlayer(player: AVPlayer(url: out))
                    .frame(maxWidth: .infinity, maxHeight: 420)
                    .cornerRadius(10)
                HStack {
                    PhotosPicker(selection: $videoItem, matching: .videos) {
                        Label("Pick", systemImage: "film")
                    }.buttonStyle(.bordered)
                    Button { video.reprocess(modelName: modelName, options: currentOptions) } label: {
                        Label("Re-process", systemImage: "arrow.clockwise")
                    }.buttonStyle(.bordered)
                    Button { video.saveToPhotos() } label: {
                        Label(video.savedToPhotos ? "Saved" : "Save",
                              systemImage: video.savedToPhotos ? "checkmark" : "square.and.arrow.down")
                    }
                    .buttonStyle(.borderedProminent)
                    .disabled(video.savedToPhotos)
                }
            } else {
                VStack(spacing: 12) {
                    Image(systemName: "film").font(.system(size: 50)).foregroundColor(.gray)
                    Text("Pick a video to segment every frame").foregroundColor(.secondary)
                    PhotosPicker(selection: $videoItem, matching: .videos) {
                        Label("Pick Video", systemImage: "film").padding(.vertical, 6)
                    }.buttonStyle(.borderedProminent)
                    Text("Same engine as Camera: tracker keeps each object's colour consistent across the clip; the Conf / Max / Resolution sliders apply (use Re-process after changing them).")
                        .font(.caption2).foregroundColor(.secondary)
                        .multilineTextAlignment(.center).padding(.horizontal, 28)
                }
            }
            if let s = video.status {
                Text(s).font(.caption).foregroundColor(.secondary)
            }
        }
        .frame(maxWidth: .infinity, maxHeight: .infinity)
        .padding()
        .onChange(of: videoItem) { item in
            guard let item else { return }
            video.status = "Loading video…"
            Task {
                if let movie = try? await item.loadTransferable(type: Movie.self) {
                    video.process(url: movie.url, modelName: modelName, options: currentOptions)
                } else {
                    video.status = "Could not load video"
                }
            }
        }
    }

    static func viewToImage(_ pt: CGPoint, viewSize: CGSize, imageSize: CGSize) -> CGPoint {
        let ia = imageSize.width / imageSize.height, va = viewSize.width / viewSize.height
        let shown: CGSize, offset: CGPoint
        if ia > va {
            let w = viewSize.width, h = w / ia
            shown = CGSize(width: w, height: h); offset = CGPoint(x: 0, y: (viewSize.height - h) / 2)
        } else {
            let h = viewSize.height, w = h * ia
            shown = CGSize(width: w, height: h); offset = CGPoint(x: (viewSize.width - w) / 2, y: 0)
        }
        let x = (pt.x - offset.x) / shown.width * imageSize.width
        let y = (pt.y - offset.y) / shown.height * imageSize.height
        return CGPoint(x: min(max(x, 0), imageSize.width), y: min(max(y, 0), imageSize.height))
    }
}

/// Owns the photo-mode FastSAM session, confined to a background queue. The model can be
/// hot-swapped when the user picks a different resolution.
final class PhotoModel: ObservableObject {
    @Published var image: UIImage?
    @Published var overlay: CGImage?
    @Published var selected: CGImage?
    @Published var isProcessing = false

    @Published private(set) var isReady: Bool = false

    var confidence: Float = 0.4
    var maxInstances: Int = 100

    private let queue = DispatchQueue(label: "fastsam.photo")
    private var session: FastSamSession?
    private var modelName: String = "FastSAM_s_512"

    private var options: FastSamSession.Options {
        FastSamSession.Options(confidenceThreshold: confidence, iouThreshold: 0.9, maxInstances: maxInstances)
    }

    init() { loadModel(name: modelName) }

    func loadModel(name: String) {
        queue.async { [weak self] in
            guard let self else { return }
            self.modelName = name
            self.session = try? FastSamSession(modelName: name,
                                               config: RuntimeConfig(computeUnits: .neuralEnginePreferred))
            let ok = self.session != nil
            DispatchQueue.main.async { self.isReady = ok }
        }
    }

    func setImage(_ ui: UIImage) {
        guard let cg = ui.cgImage else { return }
        image = ui; overlay = nil; selected = nil; isProcessing = true
        queue.async { [weak self] in
            guard let self, let session = self.session else { return }
            try? session.setImage(cg)
            let map = try? session.segmentEverythingMask(options: self.options)
            DispatchQueue.main.async { self.overlay = map; self.isProcessing = false }
        }
    }

    /// Re-run segmentation on the current image — used after a resolution swap.
    func refreshIfNeeded() {
        guard let img = image else { return }
        setImage(img)
    }

    func selectInstance(at point: CGPoint) {
        guard let session else { return }
        isProcessing = true
        queue.async { [weak self] in
            guard let self else { return }
            let instance = try? session.segment(at: point, options: self.options)
            DispatchQueue.main.async {
                self.selected = instance?.mask.cgImage
                self.isProcessing = false
            }
        }
    }

    func clearSelection() { selected = nil }
}

/// PhotosPicker video payload — copies the picked movie into a temp file we can read frames from.
struct Movie: Transferable {
    let url: URL
    static var transferRepresentation: some TransferRepresentation {
        FileRepresentation(contentType: .movie) { movie in
            SentTransferredFile(movie.url)
        } importing: { received in
            let copy = FileManager.default.temporaryDirectory
                .appendingPathComponent("fastsam_in_\(UUID().uuidString).mov")
            try? FileManager.default.removeItem(at: copy)
            try FileManager.default.copyItem(at: received.file, to: copy)
            return Movie(url: copy)
        }
    }
}

private extension UIImage {
    func fixedOrientation() -> UIImage {
        guard imageOrientation != .up else { return self }
        UIGraphicsBeginImageContextWithOptions(size, false, scale)
        draw(in: CGRect(origin: .zero, size: size))
        let normalized = UIGraphicsGetImageFromCurrentImageContext() ?? self
        UIGraphicsEndImageContext()
        return normalized
    }
}
