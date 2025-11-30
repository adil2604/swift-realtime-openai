import Foundation
import AVFAudio
@preconcurrency import LiveKitWebRTC

final class AudioRMSObserver: NSObject, LKRTCAudioRenderer, @unchecked Sendable {
	var callback: ((Float) -> Void)?

	func render(pcmBuffer: AVAudioPCMBuffer) {
		let channelCount = Int(pcmBuffer.format.channelCount)
		let length = Int(pcmBuffer.frameLength)
		
		guard length > 0, channelCount > 0 else { return }

		var sum: Float = 0
		var totalSamples = 0

		if let floatData = pcmBuffer.floatChannelData {
			for channel in 0..<channelCount {
				let data = floatData[channel]
				for i in 0..<length {
					let sample = data[i]
					sum += sample * sample
				}
			}
			totalSamples = length * channelCount
		} else if let int16Data = pcmBuffer.int16ChannelData {
			for channel in 0..<channelCount {
				let data = int16Data[channel]
				for i in 0..<length {
					let sample = Float(data[i]) / 32768.0
					sum += sample * sample
				}
			}
			totalSamples = length * channelCount
		}

		if totalSamples > 0 {
			let rms = sqrt(sum / Float(totalSamples))
			callback?(rms)
		}
	}
}
