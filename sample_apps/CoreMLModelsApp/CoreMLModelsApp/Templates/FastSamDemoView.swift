import SwiftUI
import PhotosUI
import CoreGraphics

/// FastSAM "segment everything" demo for the hub.
///
/// FastSAM is a YOLOv8-seg model (see `SAMKit/FastSAM.swift`), so unlike the SAM template it
/// segments every object in one pass and a tap just *selects* one. Models are resolved from
/// the hub's download cache (`Paths.modelDir`). Wire this up via the `"fast_sam"` template
/// case in `DemoLauncherView` — see `FASTSAM_HUB_INTEGRATION.md`.
struct FastSamDemoView: View {
    let model: ModelEntry

    @State private var item: PhotosPickerItem?
    @StateObject private var engine = FastSamEngine()

    var body: some View {
        VStack(spacing: 16) {
            ZStack {
                if let image = engine.image {
                    GeometryReader { geo in
                        Image(uiImage: image)
                            .resizable().scaledToFit()
                            .frame(width: geo.size.width, height: geo.size.height)
                            .overlay {
                                overlay(in: geo.size, image: image)
                            }
                            .contentShape(Rectangle())
                            .onTapGesture { loc in
                                let p = viewToImage(loc, viewSize: geo.size, imageSize: image.size)
                                engine.select(at: p)
                            }
                    }
                } else {
                    placeholder
                }
                if engine.isProcessing {
                    ProgressView().progressViewStyle(CircularProgressViewStyle(tint: .white)).scaleEffect(1.3)
                }
            }
            .frame(maxWidth: .infinity, maxHeight: .infinity)

            HStack {
                PhotosPicker(selection: $item, matching: .images) {
                    Label(engine.image == nil ? "Pick Photo" : "Change Photo", systemImage: "photo.badge.plus")
                }
                .buttonStyle(.bordered)
                if engine.selected != nil {
                    Button("Show all") { engine.clearSelection() }.buttonStyle(.bordered)
                }
            }
            .padding(.bottom, 12)
        }
        .task { engine.load(modelId: model.id) }
        .onChange(of: item) { _, newItem in
            guard let newItem else { return }
            Task {
                if let data = try? await newItem.loadTransferable(type: Data.self),
                   let ui = UIImage(data: data) {
                    engine.setImage(ui)
                }
            }
        }
    }

    @ViewBuilder
    private func overlay(in size: CGSize, image: UIImage) -> some View {
        if let sel = engine.selected {
            Image(uiImage: UIImage(cgImage: sel)).resizable().scaledToFit()
                .frame(width: size.width, height: size.height).opacity(0.6)
        } else if let map = engine.overlay {
            Image(uiImage: UIImage(cgImage: map)).resizable().scaledToFit()
                .frame(width: size.width, height: size.height).opacity(0.55)
        }
    }

    @ViewBuilder
    private var placeholder: some View {
        VStack(spacing: 12) {
            Image(systemName: engine.isReady ? "square.grid.3x3.fill" : "exclamationmark.triangle")
                .font(.system(size: 56)).foregroundStyle(.secondary)
            Text(engine.status).font(.callout).foregroundStyle(.secondary)
                .multilineTextAlignment(.center).padding(.horizontal, 24)
        }
    }

    private func viewToImage(_ pt: CGPoint, viewSize: CGSize, imageSize: CGSize) -> CGPoint {
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

/// Owns the FastSAM session for the hub demo, confined to a background queue.
final class FastSamEngine: ObservableObject {
    @Published var image: UIImage?
    @Published var overlay: CGImage?
    @Published var selected: CGImage?
    @Published var isProcessing = false
    @Published var status = "Loading FastSAM…"

    private let queue = DispatchQueue(label: "hub.fastsam")
    private var session: FastSamSession?

    var isReady: Bool { session != nil }

    /// Resolve FastSAM_s.mlpackage / .mlmodelc from the model's download directory.
    func load(modelId: String) {
        guard session == nil else { return }
        let dir = Paths.modelDir(id: modelId)
        let fm = FileManager.default
        let candidates = (try? fm.contentsOfDirectory(at: dir, includingPropertiesForKeys: nil)) ?? []
        guard let url = candidates.first(where: { ["mlmodelc", "mlpackage"].contains($0.pathExtension) }) else {
            status = "FastSAM model not found — download it first"
            return
        }
        queue.async {
            let ref = FastSamSession.ModelRef(detectorURL: url)
            let s = try? FastSamSession(model: ref, config: RuntimeConfig(computeUnits: .bestAvailable, enableFP16: true))
            DispatchQueue.main.async {
                self.session = s
                self.status = s == nil ? "Failed to load FastSAM" : "Pick a photo to segment everything"
            }
        }
    }

    func setImage(_ ui: UIImage) {
        guard let session, let cg = ui.cgImage else { return }
        image = ui; overlay = nil; selected = nil; isProcessing = true
        queue.async {
            try? session.setImage(cg)
            let map = try? session.segmentEverythingMask()
            DispatchQueue.main.async { self.overlay = map; self.isProcessing = false }
        }
    }

    func select(at point: CGPoint) {
        guard let session else { return }
        isProcessing = true
        queue.async {
            let instance = try? session.segment(at: point)
            DispatchQueue.main.async { self.selected = instance?.mask.cgImage; self.isProcessing = false }
        }
    }

    func clearSelection() { selected = nil }
}
