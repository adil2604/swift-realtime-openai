import Foundation
@preconcurrency import LiveKitWebRTC

final class AudioRMSObserver: NSObject, LKRTCAudioRenderer, @unchecked Sendable {
	var callback: ((Float) -> Void)?

	func renderData(_ audioData: UnsafeMutablePointer<Int16>?, numberOfFrames: Int, numberOfChannels: Int) {
		guard let audioData else { return }

		let totalSamples = numberOfFrames * numberOfChannels
		let frames = UnsafeBufferPointer(start: audioData, count: totalSamples)

		var sum: Float = 0
		for frame in frames {
			let v = Float(frame) / 32768.0
			sum += v * v
		}

		let rms = sqrt(sum / Float(totalSamples))
		callback?(rms)
	}
}

