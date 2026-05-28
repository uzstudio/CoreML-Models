import AVFoundation
import CoreImage
import CoreGraphics
import Combine
import Photos
import SAMKit

/// Offline video segmentation: read a clip frame-by-frame, run FastSAM "segment everything"
/// on each frame (tracker ON → stable per-object colours over time), composite the mask over
/// the frame, and write a new H.264 .mp4. Not real-time bound, so it can use a higher
/// resolution and more instances than the live camera.
final class VideoProcessor: ObservableObject {

    @Published var isProcessing = false
    @Published var progress: Double = 0
    @Published var previewFrame: CGImage?   // live composited frame while processing
    @Published var modelInputDebug: CGImage?  // exact frame fed to the model (no masks)
    var debugShowModelInput = false   // flip on to overlay the raw model-input frame
    @Published var outputURL: URL?
    @Published var status: String?
    @Published var savedToPhotos = false
    var hasInput: Bool { lastURL != nil }

    // Colour-managed render so the model gets proper sRGB (matching the camera). An earlier
    // attempt to disable colour management (NSNull) colour-shifted BT.709 video and *hurt*
    // detection — video frames carry a colour space and must be converted, not passed raw.
    private let ciContext = CIContext(options: [
        .workingColorSpace: CGColorSpace(name: CGColorSpace.sRGB)!,
        .outputColorSpace: CGColorSpace(name: CGColorSpace.sRGB)!,
    ])
    private let queue = DispatchQueue(label: "fastsam.video")          // orchestration (blocks on semaphores)
    private let writerQueue = DispatchQueue(label: "fastsam.video.writer")  // AVAssetWriter pull callback
    private var lastURL: URL?

    /// Mask overlay opacity in the burned-in output.
    var overlayAlpha: CGFloat = 0.55
    /// Cap the output's longest side (memory + speed). 4K clips would otherwise OOM.
    var maxOutputDimension = 1280
    /// Temporal mask smoothing — % weight of the previous frame (0 = off / crisp like the camera,
    /// higher = less flicker but softer/laggier masks). Light by default.
    var smoothingPrevWeight = 15

    /// Re-run on the same clip with (possibly changed) slider values — parity with the camera's
    /// live sliders, but applied as a re-process since video is offline.
    func reprocess(modelName: String, options: FastSamSession.Options) {
        guard let url = lastURL else { return }
        process(url: url, modelName: modelName, options: options)
    }

    func process(url: URL, modelName: String, options: FastSamSession.Options) {
        guard !isProcessing else { return }
        lastURL = url
        isProcessing = true; progress = 0; outputURL = nil; status = "Preparing…"
        savedToPhotos = false; previewFrame = nil; prevMask = nil; modelInputDebug = nil

        queue.async { [weak self] in
            guard let self else { return }
            do {
                try self.run(url: url, modelName: modelName, options: options)
            } catch {
                DispatchQueue.main.async {
                    self.status = "Failed: \(error.localizedDescription)"
                    self.isProcessing = false
                }
            }
        }
    }

    private func run(url: URL, modelName: String, options: FastSamSession.Options) throws {
        let asset = AVURLAsset(url: url)
        guard let track = asset.tracks(withMediaType: .video).first else {
            throw NSError(domain: "VideoProcessor", code: 1,
                          userInfo: [NSLocalizedDescriptionKey: "No video track"])
        }

        let session = try FastSamSession(modelName: modelName,
                                         config: RuntimeConfig(computeUnits: .neuralEnginePreferred))
        session.trackColors = true   // temporal → consistent colours

        // Reader (native BGRA frames).
        let reader = try AVAssetReader(asset: asset)
        let readerOutput = AVAssetReaderTrackOutput(
            track: track,
            outputSettings: [kCVPixelBufferPixelFormatTypeKey as String: kCVPixelFormatType_32BGRA])
        readerOutput.alwaysCopiesSampleData = false
        reader.add(readerOutput)

        // Orient frames UPRIGHT before the model. The camera path feeds upright frames (via the
        // connection's videoOrientation); video frames come in native/storage orientation, and an
        // iPhone "portrait" clip is landscape pixels + a 90° transform. Feeding those raw handed
        // the model a sideways scene → YOLOv8-seg (trained upright) failed even on obvious objects.
        // We bake the rotation in here, so the output is already upright (no writer transform).
        let xform = track.preferredTransform
        let displayRect = CGRect(origin: .zero, size: track.naturalSize).applying(xform)
        let dispW = abs(displayRect.width), dispH = abs(displayRect.height)
        // Cap the longest side (memory + speed; 4K would otherwise OOM).
        let s = min(1.0, Double(maxOutputDimension) / Double(max(dispW, dispH, 1)))
        var outW = Int((dispW * s).rounded()); outW -= outW % 2; outW = max(2, outW)
        var outH = Int((dispH * s).rounded()); outH -= outH % 2; outH = max(2, outH)
        let outURL = FileManager.default.temporaryDirectory
            .appendingPathComponent("fastsam_\(Int(Date().timeIntervalSince1970)).mp4")
        try? FileManager.default.removeItem(at: outURL)

        let writer = try AVAssetWriter(url: outURL, fileType: .mp4)
        let writerInput = AVAssetWriterInput(mediaType: .video, outputSettings: [
            AVVideoCodecKey: AVVideoCodecType.h264,
            AVVideoWidthKey: outW,
            AVVideoHeightKey: outH,
        ])
        writerInput.expectsMediaDataInRealTime = false
        // Frames are already upright → identity transform (no metadata rotation).
        let adaptor = AVAssetWriterInputPixelBufferAdaptor(
            assetWriterInput: writerInput,
            sourcePixelBufferAttributes: [
                kCVPixelBufferPixelFormatTypeKey as String: kCVPixelFormatType_32BGRA,
                kCVPixelBufferWidthKey as String: outW,
                kCVPixelBufferHeightKey as String: outH,
            ])
        writer.add(writerInput)

        // Pool for the oriented, upright model-input / composite-base buffers.
        var framePool: CVPixelBufferPool?
        CVPixelBufferPoolCreate(nil, nil, [
            kCVPixelBufferPixelFormatTypeKey as String: kCVPixelFormatType_32BGRA,
            kCVPixelBufferWidthKey as String: outW,
            kCVPixelBufferHeightKey as String: outH,
            kCVPixelBufferIOSurfacePropertiesKey as String: [:],
        ] as CFDictionary, &framePool)

        reader.startReading()
        writer.startWriting()
        writer.startSession(atSourceTime: .zero)

        let fps = max(1, Double(track.nominalFrameRate))
        let totalFrames = max(1, asset.duration.seconds * fps)
        var processed = 0

        DispatchQueue.main.async { self.status = "Segmenting…" }

        let done = DispatchSemaphore(value: 0)
        writerInput.requestMediaDataWhenReady(on: writerQueue) { [weak self] in
            guard let self else { return }
            while writerInput.isReadyForMoreMediaData {
                // Per-frame autorelease pool — without this, CMSampleBuffers / CIImage temporaries
                // / pixel buffers pile up across the long-running loop and OOM on big clips.
                let keepGoing: Bool = autoreleasepool {
                    guard reader.status == .reading,
                          let sample = readerOutput.copyNextSampleBuffer(),
                          let pb = CMSampleBufferGetImageBuffer(sample) else {
                        return false
                    }
                    let pts = CMSampleBufferGetPresentationTimeStamp(sample)

                    // 1. Orient the native frame upright + scale to output size.
                    let orientedCI = self.orient(CIImage(cvPixelBuffer: pb),
                                                 transform: xform, outW: outW, outH: outH)
                    // 2. Render it to a buffer and feed THAT (upright) to the model.
                    guard let framePB = self.render(orientedCI, pool: framePool) else { return true }
                    try? session.setImage(framePB)
                    let rawMask = try? session.segmentEverythingMask(options: options)
                    let mask = self.smooth(rawMask)   // temporal EMA → less flicker

                    // 3. Composite the mask over the (already upright) oriented frame.
                    var resultCI = orientedCI
                    if let mask {
                        resultCI = self.maskOverlay(mask, outW: outW, outH: outH).composited(over: orientedCI)
                    }
                    let outPB = self.render(resultCI, pool: adaptor.pixelBufferPool)
                    if let outPB { adaptor.append(outPB, withPresentationTime: pts) }

                    processed += 1
                    if processed % 3 == 0 {
                        let p = min(1.0, Double(processed) / totalFrames)
                        let preview = outPB.flatMap { self.cgImage(from: $0) }   // already upright
                        let dbg = self.debugShowModelInput ? self.cgImage(from: framePB) : nil
                        DispatchQueue.main.async {
                            self.progress = p
                            if let preview { self.previewFrame = preview }
                            if let dbg { self.modelInputDebug = dbg }
                        }
                    }
                    return true
                }
                if !keepGoing {
                    writerInput.markAsFinished()
                    done.signal()
                    return
                }
            }
        }
        done.wait()

        let finished = DispatchSemaphore(value: 0)
        writer.finishWriting { finished.signal() }
        finished.wait()

        if writer.status == .completed {
            DispatchQueue.main.async {
                self.progress = 1
                self.outputURL = outURL
                self.status = "Done (\(processed) frames)"
                self.isProcessing = false
            }
        } else {
            throw writer.error ?? NSError(domain: "VideoProcessor", code: 2,
                userInfo: [NSLocalizedDescriptionKey: "Writer failed"])
        }
    }

    /// Rotate/flip a native frame to upright via the track transform, then scale to outW×outH
    /// (origin normalised to 0,0).
    private func orient(_ ci: CIImage, transform: CGAffineTransform, outW: Int, outH: Int) -> CIImage {
        let oriented = ci.transformed(by: transform)
        let atOrigin = oriented.transformed(
            by: CGAffineTransform(translationX: -oriented.extent.minX, y: -oriented.extent.minY))
        let sx = CGFloat(outW) / max(1, atOrigin.extent.width)
        let sy = CGFloat(outH) / max(1, atOrigin.extent.height)
        return atOrigin.transformed(by: CGAffineTransform(scaleX: sx, y: sy))
    }

    /// The mask scaled to output size with its alpha dimmed to `overlayAlpha`.
    private func maskOverlay(_ mask: CGImage, outW: Int, outH: Int) -> CIImage {
        CIImage(cgImage: mask)
            .transformed(by: CGAffineTransform(scaleX: CGFloat(outW) / CGFloat(mask.width),
                                               y: CGFloat(outH) / CGFloat(mask.height)))
            .applyingFilter("CIColorMatrix",
                            parameters: ["inputAVector": CIVector(x: 0, y: 0, z: 0, w: overlayAlpha)])
    }

    private func render(_ ci: CIImage, pool: CVPixelBufferPool?) -> CVPixelBuffer? {
        guard let pool = pool else { return nil }
        var pb: CVPixelBuffer?
        guard CVPixelBufferPoolCreatePixelBuffer(nil, pool, &pb) == kCVReturnSuccess,
              let out = pb else { return nil }
        ciContext.render(ci, to: out)
        return out
    }

    private func cgImage(from pb: CVPixelBuffer) -> CGImage? {
        let w = CVPixelBufferGetWidth(pb), h = CVPixelBufferGetHeight(pb)
        return ciContext.createCGImage(CIImage(cvPixelBuffer: pb),
                                       from: CGRect(x: 0, y: 0, width: w, height: h))
    }

    // MARK: - Temporal mask smoothing (flicker reduction)

    private var prevMask: [UInt8]?
    private var prevMaskW = 0, prevMaskH = 0

    /// EMA-blend this frame's (small, proto-resolution) mask with the previous one. Colours are
    /// already stable via the tracker, so blending the same object's overlay just fades presence
    /// in/out over ~2-3 frames instead of blinking. Cheap (proto-region, ~128² px).
    private func smooth(_ mask: CGImage?) -> CGImage? {
        guard let mask = mask else { prevMask = nil; return nil }
        let w = mask.width, h = mask.height
        let cs = CGColorSpaceCreateDeviceRGB()
        let info = CGImageAlphaInfo.premultipliedLast.rawValue
        guard let ctx = CGContext(data: nil, width: w, height: h, bitsPerComponent: 8,
                                  bytesPerRow: w * 4, space: cs, bitmapInfo: info),
              let base = ctx.data else { return mask }
        ctx.draw(mask, in: CGRect(x: 0, y: 0, width: w, height: h))
        let cur = base.bindMemory(to: UInt8.self, capacity: w * h * 4)
        let count = w * h * 4
        let pw = max(0, min(100, smoothingPrevWeight))   // % weight of the previous frame
        if pw > 0, let prev = prevMask, prevMaskW == w, prevMaskH == h, prev.count == count {
            let cw = 100 - pw
            for i in 0..<count {
                cur[i] = UInt8((Int(cur[i]) * cw + Int(prev[i]) * pw) / 100)
            }
        }
        prevMask = Array(UnsafeBufferPointer(start: cur, count: count))
        prevMaskW = w; prevMaskH = h
        return ctx.makeImage()
    }

    func saveToPhotos() {
        guard let url = outputURL else { return }
        PHPhotoLibrary.requestAuthorization(for: .addOnly) { [weak self] status in
            guard status == .authorized || status == .limited else {
                DispatchQueue.main.async { self?.status = "Photos access denied" }
                return
            }
            PHPhotoLibrary.shared().performChanges {
                PHAssetCreationRequest.creationRequestForAssetFromVideo(atFileURL: url)
            } completionHandler: { ok, _ in
                DispatchQueue.main.async {
                    self?.savedToPhotos = ok
                    self?.status = ok ? "Saved to Photos" : "Save failed"
                }
            }
        }
    }
}
