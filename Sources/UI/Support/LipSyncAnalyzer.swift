import Foundation
import AVFoundation
import Accelerate
import LiveKitWebRTC

@objcMembers
public final class LipSyncAnalyzer: NSObject, LKRTCAudioRenderer {

    public struct Configuration: Sendable {
        public var volumeSmoothing: Float
        public var featureSmoothing: Float
        public var silenceVolumeThreshold: Float
        public var noiseRolloffThreshold: Float
        public var noiseZeroCrossingThreshold: Float
        public var roundedCentroidUpperBound: Float
        public var brightCentroidLowerBound: Float
        public var rolloffPercentile: Float
        public var maxDebugFrames: Int
        public var logMorphWeightThreshold: Float
        public var logVolumeThreshold: Float

        public init(
            volumeSmoothing: Float = 0.25,
            featureSmoothing: Float = 0.35,
            silenceVolumeThreshold: Float = 0.003,
            noiseRolloffThreshold: Float = 5_000,
            noiseZeroCrossingThreshold: Float = 0.12,
            roundedCentroidUpperBound: Float = 800,
            brightCentroidLowerBound: Float = 2_500,
            rolloffPercentile: Float = 0.85,
            maxDebugFrames: Int = 5,
            logMorphWeightThreshold: Float = 0.02,
            logVolumeThreshold: Float = 0.002
        ) {
            self.volumeSmoothing = volumeSmoothing
            self.featureSmoothing = featureSmoothing
            self.silenceVolumeThreshold = silenceVolumeThreshold
            self.noiseRolloffThreshold = noiseRolloffThreshold
            self.noiseZeroCrossingThreshold = noiseZeroCrossingThreshold
            self.roundedCentroidUpperBound = roundedCentroidUpperBound
            self.brightCentroidLowerBound = brightCentroidLowerBound
            self.rolloffPercentile = rolloffPercentile
            self.maxDebugFrames = maxDebugFrames
            self.logMorphWeightThreshold = logMorphWeightThreshold
            self.logVolumeThreshold = logVolumeThreshold
        }
    }

    private struct AudioMetrics {
        var volume: Float
        var centroid: Float
        var rolloff: Float
        var zcr: Float
    }

    private final class FFTResources {
        let frameLength: Int
        let halfLength: Int
        var sampleRate: Float {
            didSet {
                if oldValue != sampleRate {
                    updateFrequencyAxis()
                }
            }
        }

        let fft: vDSP.FFT<DSPSplitComplex>
        var window: [Float]
        var realBuffer: [Float]
        var imagBuffer: [Float]
        var magnitudes: [Float]
        var windowedBuffer: [Float]
        var weightedBuffer: [Float]
        var freqAxis: [Float]

        init?(frameLength: Int, sampleRate: Float) {
            guard frameLength > 0, frameLength.isMultiple(of: 2) else { return nil }
            let log2n = vDSP_Length(log2(Float(frameLength)))
            guard let fft = vDSP.FFT(log2n: log2n, radix: .radix2, ofType: DSPSplitComplex.self) else { return nil }

            self.frameLength = frameLength
            self.halfLength = frameLength / 2
            self.sampleRate = sampleRate
            self.fft = fft
            self.window = vDSP.window(ofType: Float.self,
                                      usingSequence: .hanningDenormalized,
                                      count: frameLength,
                                      isHalfWindow: false)
            self.realBuffer = .init(repeating: 0, count: halfLength)
            self.imagBuffer = .init(repeating: 0, count: halfLength)
            self.magnitudes = .init(repeating: 0, count: halfLength)
            self.windowedBuffer = .init(repeating: 0, count: frameLength)
            self.weightedBuffer = .init(repeating: 0, count: halfLength)
            self.freqAxis = []
            updateFrequencyAxis()
        }

        private func updateFrequencyAxis() {
            guard frameLength > 0 else {
                freqAxis = []
                return
            }
            freqAxis = (0..<halfLength).map { Float($0) * (sampleRate / Float(frameLength)) }
        }
    }

    // MARK: - Public API
    public var onMorphsUpdated: (([String: Float]) -> Void)?
    public var logMorphs: Bool = false
    public var configuration: Configuration

    // MARK: - State
    private var fftResources: FFTResources?
    private var smoothedVolume: Float = 0
    private var smoothedCentroid: Float = 0
    private var smoothedRolloff: Float = 0
    private var smoothedZCR: Float = 0
    private var didLogFormatInfo = false
    private var lastFrameLengthWarning: UInt32?
    private var debugFramePrints = 0

    public init(configuration: Configuration = .init()) {
        self.configuration = configuration
        super.init()
    }

    // MARK: - LKRTCAudioRenderer
    public func render(pcmBuffer: AVAudioPCMBuffer) {
        guard let channel = pcmBuffer.floatChannelData?[0] else {
            if logMorphs {
                print("[LipSync] Missing channel data (frameLength=\(pcmBuffer.frameLength))")
            }
            return
        }

        let frameLength = Int(pcmBuffer.frameLength)
        guard frameLength > 0 else { return }

        let sampleRate = max(Float(pcmBuffer.format.sampleRate), 1)

        guard let resources = ensureFFTResources(frameLength: frameLength, sampleRate: sampleRate) else {
            warnFrameLengthIfNeeded(length: pcmBuffer.frameLength)
            return
        }

        logFormatInfoIfNeeded(buffer: pcmBuffer)
        logDebugFrame(sampleRate: sampleRate, frameLength: frameLength)

        let samples = UnsafeBufferPointer(start: channel, count: frameLength)
        let metrics = analyze(samples: samples, resources: resources)
        let smoothedMetrics = smooth(metrics: metrics)
        let morphs = calculateMorphs(for: smoothedMetrics)

        logMorphsIfNeeded(morphs: morphs, volume: smoothedMetrics.volume)
        onMorphsUpdated?(morphs)
    }

    // MARK: - Setup helpers
    private func ensureFFTResources(frameLength: Int, sampleRate: Float) -> FFTResources? {
        if let existing = fftResources, existing.frameLength == frameLength {
            existing.sampleRate = sampleRate
            return existing
        }

        guard let resources = FFTResources(frameLength: frameLength, sampleRate: sampleRate) else {
            return nil
        }

        fftResources = resources
        if logMorphs {
            print("[LipSync] FFT configured: frameLength=\(frameLength)")
        }
        return resources
    }

    private func warnFrameLengthIfNeeded(length: UInt32) {
        guard logMorphs else { return }
        if lastFrameLengthWarning != length {
            lastFrameLengthWarning = length
            print("[LipSync] Skipping frame: FFT not configured for length \(length)")
        }
    }

    private func logFormatInfoIfNeeded(buffer: AVAudioPCMBuffer) {
        guard logMorphs, !didLogFormatInfo else { return }
        didLogFormatInfo = true
        print("[LipSync] Renderer attached. sampleRate=\(buffer.format.sampleRate), frameLength=\(buffer.frameLength)")
    }

    private func logDebugFrame(sampleRate: Float, frameLength: Int) {
        guard logMorphs, debugFramePrints < configuration.maxDebugFrames else { return }
        debugFramePrints += 1
        print("[LipSync] frame \(debugFramePrints): sampleRate=\(sampleRate) len=\(frameLength)")
    }

    // MARK: - Analysis pipeline
    private func analyze(samples: UnsafeBufferPointer<Float>, resources: FFTResources) -> AudioMetrics {
        let volume = rootMeanSquare(for: samples)
        let zcr = zeroCrossingRate(for: samples)
        performFFT(on: samples, resources: resources)
        let spectral = spectralFeatures(using: resources)

        return AudioMetrics(volume: volume,
                            centroid: spectral.centroid,
                            rolloff: spectral.rolloff,
                            zcr: zcr)
    }

    private func rootMeanSquare(for samples: UnsafeBufferPointer<Float>) -> Float {
        guard samples.count > 0 else { return 0 }
        var rms: Float = 0
        vDSP_rmsqv(samples.baseAddress!, 1, &rms, vDSP_Length(samples.count))
        return rms
    }

    private func zeroCrossingRate(for samples: UnsafeBufferPointer<Float>) -> Float {
        guard samples.count > 1 else { return 0 }
        var crossings: Float = 0
        var previous = samples[0]

        for sample in samples.dropFirst() {
            if sample * previous < 0 {
                crossings += 1
            }
            previous = sample
        }

        return crossings / Float(samples.count)
    }

    private func performFFT(on samples: UnsafeBufferPointer<Float>, resources: FFTResources) {
        vDSP.multiply(samples, resources.window, result: &resources.windowedBuffer)

        resources.realBuffer.withUnsafeMutableBufferPointer { realPtr in
            resources.imagBuffer.withUnsafeMutableBufferPointer { imagPtr in
                var split = DSPSplitComplex(realp: realPtr.baseAddress!, imagp: imagPtr.baseAddress!)

                resources.windowedBuffer.withUnsafeBufferPointer { windowPtr in
                    windowPtr.baseAddress!.withMemoryRebound(to: DSPComplex.self, capacity: resources.halfLength) { complexPtr in
                        vDSP_ctoz(complexPtr, 2, &split, 1, vDSP_Length(resources.halfLength))
                        resources.fft.forward(input: split, output: &split)
                    }
                }

                resources.magnitudes.withUnsafeMutableBufferPointer { magnitudePtr in
                    vDSP_zvabs(&split, 1, magnitudePtr.baseAddress!, 1, vDSP_Length(resources.halfLength))
                }
            }
        }
    }

    private func spectralFeatures(using resources: FFTResources) -> (centroid: Float, rolloff: Float) {
        guard !resources.freqAxis.isEmpty else { return (0, 0) }

        vDSP.multiply(resources.freqAxis, resources.magnitudes, result: &resources.weightedBuffer)

        var sumMagnitude: Float = 0
        var sumWeighted: Float = 0
        vDSP_sve(resources.magnitudes, 1, &sumMagnitude, vDSP_Length(resources.magnitudes.count))
        vDSP_sve(resources.weightedBuffer, 1, &sumWeighted, vDSP_Length(resources.weightedBuffer.count))

        let centroid = sumMagnitude > Float.ulpOfOne ? (sumWeighted / sumMagnitude) : 0
        let rolloff = spectralRolloff(magnitudes: resources.magnitudes,
                                      freqAxis: resources.freqAxis,
                                      totalEnergy: sumMagnitude)
        return (centroid, rolloff)
    }

    private func spectralRolloff(magnitudes: [Float],
                                 freqAxis: [Float],
                                 totalEnergy: Float) -> Float {
        guard totalEnergy > Float.ulpOfOne, !freqAxis.isEmpty else { return 0 }

        let threshold = totalEnergy * configuration.rolloffPercentile
        var cumulative: Float = 0

        for (index, magnitude) in magnitudes.enumerated() {
            cumulative += magnitude
            if cumulative >= threshold {
                return freqAxis[index]
            }
        }

        return freqAxis.last ?? 0
    }

    private func smooth(metrics: AudioMetrics) -> AudioMetrics {
        smoothedVolume = blend(current: smoothedVolume,
                               newValue: metrics.volume,
                               factor: configuration.volumeSmoothing)
        smoothedCentroid = blend(current: smoothedCentroid,
                                 newValue: metrics.centroid,
                                 factor: configuration.featureSmoothing)
        smoothedRolloff = blend(current: smoothedRolloff,
                                newValue: metrics.rolloff,
                                factor: configuration.featureSmoothing)
        smoothedZCR = blend(current: smoothedZCR,
                            newValue: metrics.zcr,
                            factor: configuration.featureSmoothing)

        return AudioMetrics(volume: smoothedVolume,
                            centroid: smoothedCentroid,
                            rolloff: smoothedRolloff,
                            zcr: smoothedZCR)
    }

    private func blend(current: Float, newValue: Float, factor: Float) -> Float {
        let clampedFactor = max(0, min(1, factor))
        return newValue * clampedFactor + current * (1 - clampedFactor)
    }

    // MARK: - Morph mapping
    private func calculateMorphs(for metrics: AudioMetrics) -> [String: Float] {
        var morphs: [String: Float] = [
            "AI": 0, "E": 0, "U": 0, "FV": 0,
            "MBP": 0, "ShCh": 0, "O": 0, "L": 0, "WQ": 0
        ]

        if metrics.volume < configuration.silenceVolumeThreshold {
            morphs["MBP"] = 1
            return morphs
        }

        if metrics.rolloff > configuration.noiseRolloffThreshold ||
            metrics.zcr > configuration.noiseZeroCrossingThreshold {
            morphs["FV"] = min(metrics.volume * 6, 1)
            morphs["ShCh"] = min(metrics.volume * 4, 1)
        }

        if metrics.centroid < configuration.roundedCentroidUpperBound {
            let weight = min(metrics.volume * 3, 1)
            morphs["U"] = weight
            morphs["O"] = weight * 0.7
            morphs["WQ"] = weight * 0.4
        } else if metrics.centroid < configuration.brightCentroidLowerBound {
            let weight = min(metrics.volume * 4, 1)
            morphs["AI"] = weight
        } else {
            let weight = min(metrics.volume * 5, 1)
            morphs["E"] = weight
            morphs["L"] = weight * 0.2
        }

        let sum = morphs.values.reduce(0, +)
        if sum > 1 {
            for key in morphs.keys {
                morphs[key]! /= sum
            }
        }

        return morphs
    }

    private func logMorphsIfNeeded(morphs: [String: Float], volume: Float) {
        guard logMorphs else { return }
        let maxWeight = morphs.values.max() ?? 0
        if maxWeight > configuration.logMorphWeightThreshold ||
            volume > configuration.logVolumeThreshold {
            let topMorphs = morphs
                .sorted { $0.value > $1.value }
                .prefix(5)
                .map { "\($0.key)=\(String(format: "%.2f", $0.value))" }
                .joined(separator: ", ")
            let rmsdB = 20 * log10(max(volume, 0.0001))
            print("[LipSync] RMS=\(String(format: "%.1f", rmsdB))dB | \(topMorphs)")
        }
    }
}

