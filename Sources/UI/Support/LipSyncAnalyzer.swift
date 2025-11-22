import Foundation
import AVFoundation
import Accelerate
import LiveKitWebRTC

@objcMembers
public final class LipSyncAnalyzer: NSObject, LKRTCAudioRenderer {

    // MARK: - Public callback
    public var onMorphsUpdated: (([String: Float]) -> Void)?
    /// Set to `true` to print morph weights each frame (for debugging only).
    public var logMorphs: Bool = false
    private var didLogFormatInfo = false
    private var lastFrameLengthWarning: UInt32?
    private var debugFramePrints = 0

    // MARK: - FFT setup
    // Dynamically updated based on buffer
    private var sampleRate: Float = 48_000
    private var frameLength: Int = 480

    private var fftSetup: vDSP.FFT<DSPSplitComplex>?
    private var window: [Float] = []
    private var realBuffer: [Float] = []
    private var imagBuffer: [Float] = []
    private var magnitudes: [Float] = []
    private var windowedBuffer: [Float] = []
    private var weightedBuffer: [Float] = []
    private var freqAxis: [Float] = []

    // MARK: - EMA smoothing
    private var smoothedVolume: Float = 0
    private var smoothedCentroid: Float = 0
    private var smoothedRolloff: Float = 0
    private var smoothedZCR: Float = 0

    private let emaFactor: Float = 0.35

    public override init() {
        super.init()
        setupFFT(frameLength: frameLength)
    }

    // MARK: - FFT init
    private func setupFFT(frameLength: Int) {
        guard frameLength > 0, frameLength.isMultiple(of: 2) else {
            fftSetup = nil
            return
        }

        let log2n = vDSP_Length(log2(Float(frameLength)))
        fftSetup = vDSP.FFT(log2n: log2n, radix: .radix2, ofType: DSPSplitComplex.self)

        window = vDSP.window(ofType: Float.self,
                             usingSequence: .hanningDenormalized,
                             count: frameLength,
                             isHalfWindow: false)

        let halfLength = frameLength / 2
        realBuffer = .init(repeating: 0, count: halfLength)
        imagBuffer = .init(repeating: 0, count: halfLength)
        magnitudes = .init(repeating: 0, count: halfLength)
        windowedBuffer = .init(repeating: 0, count: frameLength)
        weightedBuffer = .init(repeating: 0, count: halfLength)
        freqAxis = (0..<halfLength).map { Float($0) * (sampleRate / Float(frameLength)) }

        self.frameLength = frameLength
        if logMorphs {
            print("[LipSync] FFT configured: frameLength=\(frameLength)")
        }
    }

    // MARK: - Main render
    public func render(pcmBuffer: AVAudioPCMBuffer) {
        // Check for sample rate change (rare but possible)
        let bufferSampleRate = Float(pcmBuffer.format.sampleRate)
        if bufferSampleRate != self.sampleRate && bufferSampleRate > 0 {
             self.sampleRate = bufferSampleRate
             // Recalculate frequency axis
             let halfLength = frameLength / 2
             freqAxis = (0..<halfLength).map { Float($0) * (sampleRate / Float(frameLength)) }
        }

        guard let channel = pcmBuffer.floatChannelData?[0] else {
            if logMorphs {
                print("[LipSync] Missing channel data (frameLength=\(pcmBuffer.frameLength))")
            }
            return
        }

        let currentFrameLength = Int(pcmBuffer.frameLength)
        if currentFrameLength != frameLength || fftSetup == nil {
            setupFFT(frameLength: currentFrameLength)
        }
        guard currentFrameLength == frameLength, let fftSetup else {
            if logMorphs, lastFrameLengthWarning != pcmBuffer.frameLength {
                lastFrameLengthWarning = pcmBuffer.frameLength
                print("[LipSync] Skipping frame: FFT not configured for length \(currentFrameLength)")
            }
            return
        }

        if logMorphs, !didLogFormatInfo {
            didLogFormatInfo = true
            print("[LipSync] Renderer attached. sampleRate=\(pcmBuffer.format.sampleRate), frameLength=\(pcmBuffer.frameLength)")
        }

        let samples = UnsafeBufferPointer(start: channel, count: frameLength)

        if logMorphs && debugFramePrints < 5 {
            debugFramePrints += 1
            print("[LipSync] frame \(debugFramePrints): sampleRate=\(pcmBuffer.format.sampleRate) len=\(frameLength)")
        }

        // ---- 1. RMS loudness ----
        var rms: Float = 0
        vDSP_rmsqv(samples.baseAddress!, 1, &rms, vDSP_Length(frameLength))

        // smooth volume
        smoothedVolume = rms * 0.25 + smoothedVolume * 0.75

        // ---- 2. Zero Crossing Rate (ZCR) ----
        var zcrCount: Float = 0
        // Use unsafe pointer for speed (no bound checks)
        for i in 1..<frameLength {
            if samples[i] * samples[i-1] < 0 {
                zcrCount += 1
            }
        }
        let zcr = zcrCount / Float(frameLength)
        smoothedZCR = zcr * emaFactor + smoothedZCR * (1 - emaFactor)

        // ---- 3. FFT → Magnitudes ----
        // Use pre-allocated windowedBuffer
        vDSP.multiply(samples, window, result: &windowedBuffer)

        windowedBuffer.withUnsafeBufferPointer { ptr in
            ptr.baseAddress!.withMemoryRebound(to: DSPComplex.self, capacity: frameLength / 2) { complexPtr in
                var split = DSPSplitComplex(realp: &realBuffer, imagp: &imagBuffer)
                // Important: vDSP_ctoz expects stride 2 for input (Interleaved Complex)
                vDSP_ctoz(complexPtr, 2, &split, 1, vDSP_Length(frameLength / 2))
                fftSetup.forward(input: split, output: &split)
            }
        }
        
        // Get magnitudes |v|
        realBuffer.withUnsafeMutableBufferPointer { realPtr in
            imagBuffer.withUnsafeMutableBufferPointer { imagPtr in
                magnitudes.withUnsafeMutableBufferPointer { magPtr in
                    var split = DSPSplitComplex(realp: realPtr.baseAddress!, imagp: imagPtr.baseAddress!)
                    vDSP_zvabs(&split, 1, magPtr.baseAddress!, 1, vDSP_Length(magnitudes.count))
                }
            }
        }

        // ---- 4. Spectral centroid ----
        // Use pre-allocated weightedBuffer
        vDSP.multiply(freqAxis, magnitudes, result: &weightedBuffer)

        var sumMag: Float = 0
        var sumWeighted: Float = 0
        vDSP_sve(magnitudes, 1, &sumMag, vDSP_Length(magnitudes.count))
        vDSP_sve(weightedBuffer, 1, &sumWeighted, vDSP_Length(weightedBuffer.count))

        let centroid = (sumMag > 0.0001) ? (sumWeighted / sumMag) : 0
        smoothedCentroid = centroid * emaFactor + smoothedCentroid * (1 - emaFactor)

        // ---- 5. Spectral rolloff (85%) ----
        // Loop is fine here as array is short (240 elements)
        let rolloffThreshold = sumMag * 0.85
        var cumulative: Float = 0
        var rolloffFreq: Float = 0
        for i in 0..<magnitudes.count {
            cumulative += magnitudes[i]
            if cumulative >= rolloffThreshold {
                rolloffFreq = freqAxis[i]
                break
            }
        }
        smoothedRolloff = rolloffFreq * emaFactor + smoothedRolloff * (1 - emaFactor)

        // ---- 6. Map to morphs ----
        let morphs = calculateMorphs(
            volume: smoothedVolume,
            centroid: smoothedCentroid,
            rolloff: smoothedRolloff,
            zcr: smoothedZCR
        )
        
        if logMorphs {
            let maxWeight = morphs.values.max() ?? 0
            if maxWeight > 0.02 || smoothedVolume > 0.002 {
                let topMorphs = morphs
                    .sorted { $0.value > $1.value }
                    .prefix(5)
                    .map { "\($0.key)=\(String(format: "%.2f", $0.value))" }
                    .joined(separator: ", ")
                let rmsdB = 20 * log10(max(smoothedVolume, 0.0001))
                print("[LipSync] RMS=\(String(format: "%.1f", rmsdB))dB | \(topMorphs)")
            }
        }

        onMorphsUpdated?(morphs)
    }

    // MARK: - Morph mapping
    private func calculateMorphs(volume: Float,
                                 centroid: Float,
                                 rolloff: Float,
                                 zcr: Float) -> [String: Float] {

        var m: [String: Float] = [
            "AI": 0, "E": 0, "U": 0, "FV": 0,
            "MBP": 0, "ShCh": 0, "O": 0, "L": 0, "WQ": 0
        ]

        // ========= MBP ============
        if volume < 0.003 {
            m["MBP"] = 1
            return m
        }

        // ========= FV / ShCh (noise-based) ============
        if rolloff > 5000 || zcr > 0.12 {
            m["FV"] = min(volume * 6, 1)
            m["ShCh"] = min(volume * 4, 1)
        }

        // ========= U / O (rounded vowels) ============
        if centroid < 800 {
            let w = min(volume * 3, 1)
            m["U"] = w
            m["O"] = w * 0.7
            m["WQ"] = w * 0.4
        }

        // ========= E / AI / L (mid-high vowels) ============
        if centroid >= 800 && centroid < 2500 {
            let w = min(volume * 4, 1)
            m["AI"] = w
            m["O"] = m["O"] ?? 0
        }

        if centroid >= 2500 {
            let w = min(volume * 5, 1)
            m["E"] = w
            m["L"] = 0.2 * w
        }

        // Normalization: keep sum <= 1
        let sum = m.values.reduce(0, +)
        if sum > 1 {
            for key in m.keys {
                m[key]! /= sum
            }
        }

        return m
    }
}

