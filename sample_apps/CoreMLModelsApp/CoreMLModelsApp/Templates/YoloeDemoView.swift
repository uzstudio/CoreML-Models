import SwiftUI
import PhotosUI
import CoreML
import Accelerate

/// YOLOE open-vocabulary **detection + instance segmentation**: photo + text query.
///
/// Unlike YOLO-World (which bakes the text prompt into the class head), the YOLOE
/// detector is text-free and emits per-anchor **region embeddings**; the region–text
/// similarity (YOLOE's per-scale BNContrastiveHead, folded into a 513-d augmented dot
/// product) runs here in Swift against cached text embeddings, so the vocabulary is
/// free to change. Text path: MobileCLIP → RepRTA → L2-normalize → append 1.0.
///
/// Detector: `image`[1,3,640,640] → `boxes`[1,4,8400], `region_embeddings`[1,513,8400],
///           `mask_coeffs`[1,32,8400], `mask_protos`[1,32,160,160].
struct YoloeDemoView: View {
    let model: ModelEntry

    @State private var inputImage: UIImage?
    @State private var annotatedImage: UIImage?
    @State private var queryText = "person, car, dog"
    @State private var detections: [Det] = []
    @State private var isProcessing = false
    @State private var status = ""
    @State private var processingTime: Double?
    @State private var item: PhotosPickerItem?
    @State private var confidenceThreshold: Float = 0.15
    @State private var showMasks = true
    @State private var maxDet = 50   // cap on returned detections (1...100)
    @StateObject private var session = ModelSession<(detector: MLModel, reprta: MLModel, textEnc: MLModel)>()

    struct Det: Identifiable {
        let id = UUID()
        let label: String
        let confidence: Float
        let box: CGRect        // normalized 0..1, origin top-left
        let anchor: Int
        let classIndex: Int
    }

    private let inputSize = 640
    private let numAnchors = 8400
    private let embedDim = 512
    private let augDim = 513
    private let reprtaSlots = 80
    private let nm = 32        // mask channels
    private let protoSize = 160

    private static let tokenizer: CLIPTokenizer? = {
        guard let url = Bundle.main.url(forResource: "clip_vocab", withExtension: "json") else { return nil }
        return try? CLIPTokenizer(vocabularyURL: url)
    }()

    private let colors: [UIColor] = [.systemRed, .systemBlue, .systemGreen, .systemOrange,
                                     .systemPurple, .systemYellow, .systemPink, .systemCyan]

    var body: some View {
        VStack(spacing: 0) {
            ZStack {
                if let img = annotatedImage ?? inputImage {
                    Image(uiImage: img).resizable().aspectRatio(contentMode: .fit)
                        .clipShape(RoundedRectangle(cornerRadius: 8))
                } else {
                    VStack(spacing: 12) {
                        Image(systemName: "text.viewfinder").font(.system(size: 60)).foregroundStyle(.secondary)
                        Text("Select a photo and enter object names").foregroundStyle(.secondary)
                    }
                }
            }
            .frame(maxWidth: .infinity, maxHeight: .infinity)
            .padding(.horizontal)

            if !detections.isEmpty {
                ScrollView(.horizontal, showsIndicators: false) {
                    HStack(spacing: 8) {
                        ForEach(detections) { d in
                            Text("\(d.label) \(Int(d.confidence * 100))%")
                                .font(.caption2).padding(.horizontal, 8).padding(.vertical, 4)
                                .background(.ultraThinMaterial).clipShape(Capsule())
                        }
                    }.padding(.horizontal)
                }.frame(height: 40)
            }

            VStack(spacing: 12) {
                TextField("Objects (comma-separated)", text: $queryText)
                    .textFieldStyle(.roundedBorder).font(.callout)

                HStack {
                    Text("Confidence").font(.caption2).foregroundStyle(.secondary)
                    Slider(value: $confidenceThreshold, in: 0.05...0.9)
                    Text(String(format: "%.0f%%", confidenceThreshold * 100))
                        .font(.caption2.monospacedDigit()).foregroundStyle(.secondary).frame(width: 36)
                    Menu {
                        Picker("Max detections", selection: $maxDet) {
                            ForEach([25, 50, 75, 100], id: \.self) { Text("\($0)").tag($0) }
                        }
                    } label: {
                        Text("≤\(maxDet)").font(.caption2.monospacedDigit())
                            .padding(.horizontal, 6).padding(.vertical, 2)
                            .background(.ultraThinMaterial, in: Capsule())
                    }
                    Toggle("Masks", isOn: $showMasks).labelsHidden()
                    Image(systemName: "theatermasks").font(.caption2).foregroundStyle(.secondary)
                }

                HStack {
                    TimingsLabel(loadSec: session.loadTimeSec, inferSec: processingTime)
                    if !detections.isEmpty {
                        Text("· \(detections.count)").font(.caption.monospacedDigit()).foregroundStyle(.secondary)
                    }
                    Spacer()
                    if isProcessing { ProgressView().controlSize(.small) }
                }

                HStack(spacing: 12) {
                    PhotosPicker(selection: $item, matching: .images) {
                        Label("Photo", systemImage: "photo.badge.plus")
                    }.buttonStyle(.bordered)

                    Button {
                        if let img = inputImage { Task { await runDetection(on: img) } }
                    } label: {
                        Label("Detect", systemImage: "sparkle.magnifyingglass").frame(maxWidth: .infinity)
                    }
                    .buttonStyle(.borderedProminent)
                    .disabled(isProcessing || inputImage == nil || queryText.trimmingCharacters(in: .whitespaces).isEmpty)

                    if let annotated = annotatedImage {
                        Button { UIImageWriteToSavedPhotosAlbum(annotated, nil, nil, nil) } label: {
                            Image(systemName: "arrow.down.to.line")
                        }.buttonStyle(.bordered)
                    }
                }
            }
            .padding()
        }
        .task {
            session.ensure {
                func file(_ needles: [String]) -> String? {
                    model.files.first { f in needles.contains { f.name.lowercased().contains($0) } }?.name
                }
                let detName = file(["detector"]) ?? model.files[0].name
                let reprtaName = file(["reprta"]) ?? detName
                let textName = file(["mobileclip", "clip", "text"]) ?? detName
                let det = try await ModelLoader.load(for: model, named: detName)
                let rep = try await ModelLoader.load(for: model, named: reprtaName)
                let txt = try await ModelLoader.load(for: model, named: textName)
                return (detector: det, reprta: rep, textEnc: txt)
            }
        }
        .onChange(of: item) { _, _ in loadPhoto() }
    }

    private func loadPhoto() {
        guard let item else { return }
        Task {
            if let data = try? await item.loadTransferable(type: Data.self), let img = UIImage(data: data) {
                await MainActor.run { inputImage = img; annotatedImage = nil; detections = [] }
            }
        }
    }

    // MARK: - Detection

    private func runDetection(on image: UIImage) async {
        let classes = queryText.split(separator: ",").map { $0.trimmingCharacters(in: .whitespaces) }.filter { !$0.isEmpty }
        guard !classes.isEmpty else { return }

        isProcessing = true
        do {
            status = session.loadTimeSec == nil ? "Loading models…" : "Preparing…"
            let (detector, reprta, textEnc) = try await session.get()
            guard let cgImage = ImageUtils.normalizeOrientation(image) else {
                isProcessing = false; status = "Image error"; return
            }
            let start = CFAbsoluteTimeGetCurrent()

            // 1. Text → query' [n, 513] = [normalize(reprta(MobileCLIP)), 1.0]
            status = "Encoding text…"
            let (queryPrime, n) = try encodeQueries(classes, textEnc: textEnc, reprta: reprta)
            guard n > 0 else { isProcessing = false; status = "Text encode failed"; return }

            // 2. Preprocess image (letterbox to 640)
            status = "Detecting…"
            let imgW = cgImage.width, imgH = cgImage.height
            let scale = Float(inputSize) / Float(max(imgW, imgH))
            let scaledW = Int(Float(imgW) * scale), scaledH = Int(Float(imgH) * scale)
            let padX = (inputSize - scaledW) / 2, padY = (inputSize - scaledH) / 2
            let imageTensor = try preprocessImage(cgImage, scaledW: scaledW, scaledH: scaledH, padX: padX, padY: padY)

            // 3. Detector
            let input = try MLDictionaryFeatureProvider(dictionary: ["image": imageTensor])
            let output = try await detector.prediction(from: input)
            guard let boxesMA = output.featureValue(for: "boxes")?.multiArrayValue,
                  let regionMA = output.featureValue(for: "region_embeddings")?.multiArrayValue else {
                isProcessing = false; status = "No detection output"; return
            }
            let boxes = readMatrix2D(boxesMA)        // [4, 8400]
            let region = readMatrix2D(regionMA)      // [513, 8400]

            // 4. scores [n, 8400] = query'[n,513] x region'[513,8400]
            var logits = [Float](repeating: 0, count: n * numAnchors)
            queryPrime.withUnsafeBufferPointer { aP in
                region.withUnsafeBufferPointer { bP in
                    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                                Int32(n), Int32(numAnchors), Int32(augDim),
                                1.0, aP.baseAddress, Int32(augDim),
                                bP.baseAddress, Int32(numAnchors),
                                0.0, &logits, Int32(numAnchors))
                }
            }

            // 5. Threshold + decode + NMS
            let t = confidenceThreshold
            let logitThresh: Float = (t > 0 && t < 1) ? log(t / (1 - t)) : -.greatestFiniteMagnitude
            let invW = 1.0 / (Float(imgW) * scale), invH = 1.0 / (Float(imgH) * scale)
            var all: [(CGRect, Float, Int, Int)] = []
            for k in 0..<n {
                let off = k * numAnchors
                for a in 0..<numAnchors where logits[off + a] >= logitThresh {
                    let score = 1.0 / (1.0 + exp(-logits[off + a]))
                    let cx = boxes[a], cy = boxes[numAnchors + a]
                    let bw = boxes[2 * numAnchors + a], bh = boxes[3 * numAnchors + a]
                    let nx = (cx - bw / 2 - Float(padX)) * invW, ny = (cy - bh / 2 - Float(padY)) * invH
                    let rect = CGRect(x: CGFloat(max(0, min(1, nx))), y: CGFloat(max(0, min(1, ny))),
                                      width: CGFloat(max(0, min(1, bw * invW))), height: CGFloat(max(0, min(1, bh * invH))))
                    all.append((rect, score, k, a))
                }
            }
            all.sort { $0.1 > $1.1 }
            var kept: [Int] = []
            for i in all.indices {
                var suppress = false
                for ki in kept where all[i].2 == all[ki].2 {
                    if iou(all[i].0, all[ki].0) > 0.5 { suppress = true; break }
                }
                if !suppress { kept.append(i) }
            }
            let dets = kept.prefix(max(1, min(100, maxDet))).map { i in
                Det(label: classes[all[i].2], confidence: all[i].1, box: all[i].0, anchor: all[i].3, classIndex: all[i].2)
            }

            // 6. Masks (optional)
            var maskImg: CGImage?
            if showMasks,
               let coeffsMA = output.featureValue(for: "mask_coeffs")?.multiArrayValue,
               let protosMA = output.featureValue(for: "mask_protos")?.multiArrayValue {
                maskImg = buildMask(Array(dets), coeffs: readMatrix2D(coeffsMA), protos: readProtos(protosMA),
                                    padX: padX, padY: padY, scale: scale, imgW: imgW, imgH: imgH)
            }

            let elapsed = CFAbsoluteTimeGetCurrent() - start
            let annotated = draw(Array(dets), maskImage: maskImg, on: image)
            await MainActor.run {
                detections = Array(dets); annotatedImage = annotated
                processingTime = elapsed; isProcessing = false; status = ""
            }
        } catch {
            await MainActor.run { isProcessing = false; status = "Error: \(error.localizedDescription)" }
        }
    }

    // MARK: - Text encoding (MobileCLIP -> RepRTA -> normalize, append 1.0)

    private func encodeQueries(_ queries: [String], textEnc: MLModel, reprta: MLModel) throws -> ([Float], Int) {
        guard let tokenizer = Self.tokenizer else { return ([], 0) }
        let n = min(queries.count, reprtaSlots)
        let ctxLen = tokenizer.contextLength

        let rawTpe = try MLMultiArray(shape: [1, reprtaSlots as NSNumber, embedDim as NSNumber], dataType: .float32)
        let rawPtr = rawTpe.dataPointer.assumingMemoryBound(to: Float.self)
        memset(rawPtr, 0, reprtaSlots * embedDim * MemoryLayout<Float>.size)

        for i in 0..<n {
            let tokenArr = try MLMultiArray(shape: [1, ctxLen as NSNumber], dataType: .int32)
            let tPtr = tokenArr.dataPointer.assumingMemoryBound(to: Int32.self)
            let tokens = tokenizer.tokenize(queries[i])
            for j in 0..<ctxLen { tPtr[j] = Int32(tokens[j]) }
            let out = try textEnc.prediction(from: try MLDictionaryFeatureProvider(dictionary: ["text": tokenArr]))
            guard let embMA = out.featureValue(for: "final_emb_1")?.multiArrayValue else { continue }
            var emb = Array(ImageUtils.extractFloats(embMA).prefix(embedDim))
            if emb.count < embedDim { emb.append(contentsOf: [Float](repeating: 0, count: embedDim - emb.count)) }
            var norm: Float = 0; vDSP_svesq(emb, 1, &norm, vDSP_Length(embedDim)); norm = sqrt(norm)
            if norm > 1e-8 { for j in 0..<embedDim { rawPtr[i * embedDim + j] = emb[j] / norm } }
        }

        let rOut = try reprta.prediction(from: try MLDictionaryFeatureProvider(dictionary: ["raw_tpe": rawTpe]))
        guard let tpeMA = rOut.featureValue(for: "tpe")?.multiArrayValue else { return ([], 0) }
        let tpe = ImageUtils.extractFloats(tpeMA)

        var q = [Float](repeating: 0, count: n * augDim)
        for i in 0..<n {
            let off = i * embedDim
            var norm: Float = 0
            tpe.withUnsafeBufferPointer { vDSP_svesq($0.baseAddress! + off, 1, &norm, vDSP_Length(embedDim)) }
            let invv: Float = norm > 1e-8 ? 1.0 / sqrt(norm) : 0
            for c in 0..<embedDim { q[i * augDim + c] = tpe[off + c] * invv }
            q[i * augDim + embedDim] = 1.0
        }
        return (q, n)
    }

    // MARK: - Preprocess (letterbox 0.5 gray)

    private func preprocessImage(_ cgImage: CGImage, scaledW: Int, scaledH: Int, padX: Int, padY: Int) throws -> MLMultiArray {
        guard let ctx = CGContext(data: nil, width: inputSize, height: inputSize, bitsPerComponent: 8,
                                  bytesPerRow: inputSize * 4, space: CGColorSpaceCreateDeviceRGB(),
                                  bitmapInfo: CGImageAlphaInfo.noneSkipLast.rawValue) else {
            throw NSError(domain: "Preprocess", code: 1)
        }
        ctx.setFillColor(gray: 0.5, alpha: 1.0)
        ctx.fill(CGRect(x: 0, y: 0, width: inputSize, height: inputSize))
        ctx.draw(cgImage, in: CGRect(x: padX, y: padY, width: scaledW, height: scaledH))
        guard let pixels = ctx.data else { throw NSError(domain: "Preprocess", code: 2) }
        let arr = try MLMultiArray(shape: [1, 3, inputSize as NSNumber, inputSize as NSNumber], dataType: .float32)
        let dst = arr.dataPointer.assumingMemoryBound(to: Float.self)
        let src = pixels.assumingMemoryBound(to: UInt8.self)
        let hw = inputSize * inputSize, inv: Float = 1.0 / 255.0
        for i in 0..<hw {
            dst[0 * hw + i] = Float(src[i * 4 + 0]) * inv
            dst[1 * hw + i] = Float(src[i * 4 + 1]) * inv
            dst[2 * hw + i] = Float(src[i * 4 + 2]) * inv
        }
        return arr
    }

    // MARK: - Combined mask (one BLAS matmul, proto-res, de-letterboxed)

    private func buildMask(_ dets: [Det], coeffs: [Float], protos: [Float],
                           padX: Int, padY: Int, scale: Float, imgW: Int, imgH: Int) -> CGImage? {
        let count = dets.count
        guard count > 0 else { return nil }
        let hw = protoSize * protoSize
        var a = [Float](repeating: 0, count: count * nm)
        for (i, d) in dets.enumerated() { for k in 0..<nm { a[i * nm + k] = coeffs[k * numAnchors + d.anchor] } }
        var comb = [Float](repeating: 0, count: count * hw)
        a.withUnsafeBufferPointer { aP in protos.withUnsafeBufferPointer { bP in
            cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, Int32(count), Int32(hw), Int32(nm),
                        1.0, aP.baseAddress, Int32(nm), bP.baseAddress, Int32(hw), 0.0, &comb, Int32(hw))
        } }
        var px = [UInt8](repeating: 0, count: hw * 4)
        let s = Float(protoSize) / Float(inputSize)
        for i in dets.indices.sorted(by: { dets[$0].confidence < dets[$1].confidence }) {
            let r = dets[i].box
            let x0 = Int((Float(r.minX) * Float(imgW) * scale + Float(padX)) * s)
            let y0 = Int((Float(r.minY) * Float(imgH) * scale + Float(padY)) * s)
            let x1 = Int((Float(r.maxX) * Float(imgW) * scale + Float(padX)) * s)
            let y1 = Int((Float(r.maxY) * Float(imgH) * scale + Float(padY)) * s)
            let bx0 = max(0, min(protoSize - 1, x0)), bx1 = max(0, min(protoSize - 1, x1))
            let by0 = max(0, min(protoSize - 1, y0)), by1 = max(0, min(protoSize - 1, y1))
            guard bx1 >= bx0, by1 >= by0 else { continue }
            var cr: CGFloat = 0, cg: CGFloat = 0, cb: CGFloat = 0, ca: CGFloat = 0
            colors[dets[i].classIndex % colors.count].getRed(&cr, green: &cg, blue: &cb, alpha: &ca)
            let base = i * hw
            for y in by0...by1 { let row = y * protoSize
                for x in bx0...bx1 where comb[base + row + x] > 0 {
                    let o = (row + x) * 4
                    px[o] = UInt8(cr * 255); px[o + 1] = UInt8(cg * 255); px[o + 2] = UInt8(cb * 255); px[o + 3] = 255
                }
            }
        }
        let cs = CGColorSpaceCreateDeviceRGB()
        let info = CGBitmapInfo(rawValue: CGImageAlphaInfo.premultipliedLast.rawValue)
        guard let provider = CGDataProvider(data: Data(px) as CFData),
              let full = CGImage(width: protoSize, height: protoSize, bitsPerComponent: 8, bitsPerPixel: 32,
                                 bytesPerRow: protoSize * 4, space: cs, bitmapInfo: info, provider: provider,
                                 decode: nil, shouldInterpolate: false, intent: .defaultIntent) else { return nil }
        let cx = Int((Float(padX) * s).rounded()), cy = Int((Float(padY) * s).rounded())
        let cw = Int((Float(imgW) * scale * s).rounded()), ch = Int((Float(imgH) * scale * s).rounded())
        let crop = CGRect(x: cx, y: cy, width: max(1, min(cw, protoSize - cx)), height: max(1, min(ch, protoSize - cy)))
        return full.cropping(to: crop) ?? full
    }

    // MARK: - Draw

    private func draw(_ dets: [Det], maskImage: CGImage?, on image: UIImage) -> UIImage? {
        guard let cgImage = ImageUtils.normalizeOrientation(image) else { return image }
        let w = CGFloat(cgImage.width), h = CGFloat(cgImage.height)
        return UIGraphicsImageRenderer(size: CGSize(width: w, height: h)).image { ctx in
            UIImage(cgImage: cgImage).draw(in: CGRect(x: 0, y: 0, width: w, height: h))
            if let m = maskImage {
                ctx.cgContext.saveGState(); ctx.cgContext.setAlpha(0.5)
                UIImage(cgImage: m).draw(in: CGRect(x: 0, y: 0, width: w, height: h))
                ctx.cgContext.restoreGState()
            }
            for d in dets {
                let color = colors[d.classIndex % colors.count]
                let rect = CGRect(x: d.box.minX * w, y: d.box.minY * h, width: d.box.width * w, height: d.box.height * h)
                ctx.cgContext.setStrokeColor(color.cgColor)
                ctx.cgContext.setLineWidth(max(2, w / 300))
                ctx.cgContext.stroke(rect)
                let attrs: [NSAttributedString.Key: Any] = [
                    .font: UIFont.boldSystemFont(ofSize: max(12, w / 50)),
                    .foregroundColor: UIColor.white,
                    .backgroundColor: color.withAlphaComponent(0.8)]
                ("\(d.label) \(Int(d.confidence * 100))%" as NSString)
                    .draw(at: CGPoint(x: rect.minX + 2, y: max(0, rect.minY - max(14, w / 45))), withAttributes: attrs)
            }
        }
    }

    // MARK: - Stride-aware reads (ANE pads rows; never read large FP16 tensors flat)

    private func readMatrix2D(_ a: MLMultiArray) -> [Float] {
        let shape = a.shape.map { $0.intValue }, strides = a.strides.map { $0.intValue }
        let rows = shape[shape.count - 2], cols = shape[shape.count - 1]
        let rowStride = strides[strides.count - 2], colStride = strides[strides.count - 1]
        var out = [Float](repeating: 0, count: rows * cols)
        if colStride == 1 {
            out.withUnsafeMutableBufferPointer { dst in
                if a.dataType == .float16 {
                    var s = vImage_Buffer(data: a.dataPointer, height: vImagePixelCount(rows),
                                          width: vImagePixelCount(cols), rowBytes: rowStride * 2)
                    var d = vImage_Buffer(data: UnsafeMutableRawPointer(dst.baseAddress!), height: vImagePixelCount(rows),
                                          width: vImagePixelCount(cols), rowBytes: cols * 4)
                    vImageConvert_Planar16FtoPlanarF(&s, &d, vImage_Flags(0))
                } else {
                    let src = a.dataPointer.assumingMemoryBound(to: Float.self)
                    for r in 0..<rows { memcpy(dst.baseAddress! + r * cols, src + r * rowStride, cols * 4) }
                }
            }
        } else {
            if a.dataType == .float16 {
                let p = a.dataPointer.assumingMemoryBound(to: Float16.self)
                for r in 0..<rows { for c in 0..<cols { out[r * cols + c] = Float(p[r * rowStride + c * colStride]) } }
            } else {
                let p = a.dataPointer.assumingMemoryBound(to: Float.self)
                for r in 0..<rows { for c in 0..<cols { out[r * cols + c] = p[r * rowStride + c * colStride] } }
            }
        }
        return out
    }

    private func readProtos(_ a: MLMultiArray) -> [Float] {
        let shape = a.shape.map { $0.intValue }, strides = a.strides.map { $0.intValue }
        let c = shape[1], h = shape[2], w = shape[3]
        let sc = strides[1], sh = strides[2], sw = strides[3]
        var out = [Float](repeating: 0, count: c * h * w)
        if sw == 1 && sh == w {
            out.withUnsafeMutableBufferPointer { dst in
                if a.dataType == .float16 {
                    var s = vImage_Buffer(data: a.dataPointer, height: vImagePixelCount(c),
                                          width: vImagePixelCount(h * w), rowBytes: sc * 2)
                    var d = vImage_Buffer(data: UnsafeMutableRawPointer(dst.baseAddress!), height: vImagePixelCount(c),
                                          width: vImagePixelCount(h * w), rowBytes: h * w * 4)
                    vImageConvert_Planar16FtoPlanarF(&s, &d, vImage_Flags(0))
                } else {
                    let src = a.dataPointer.assumingMemoryBound(to: Float.self)
                    for ch in 0..<c { memcpy(dst.baseAddress! + ch * h * w, src + ch * sc, h * w * 4) }
                }
            }
        } else if a.dataType == .float16 {
            let p = a.dataPointer.assumingMemoryBound(to: Float16.self)
            for ch in 0..<c { for y in 0..<h { let base = ch * sc + y * sh, o = (ch * h + y) * w
                for x in 0..<w { out[o + x] = Float(p[base + x * sw]) } } }
        } else {
            let p = a.dataPointer.assumingMemoryBound(to: Float.self)
            for ch in 0..<c { for y in 0..<h { let base = ch * sc + y * sh, o = (ch * h + y) * w
                for x in 0..<w { out[o + x] = p[base + x * sw] } } }
        }
        return out
    }

    private func iou(_ a: CGRect, _ b: CGRect) -> Float {
        let ix = max(0, min(a.maxX, b.maxX) - max(a.minX, b.minX))
        let iy = max(0, min(a.maxY, b.maxY) - max(a.minY, b.minY))
        let inter = Float(ix * iy)
        let uni = Float(a.width * a.height) + Float(b.width * b.height) - inter
        return uni > 0 ? inter / uni : 0
    }
}
