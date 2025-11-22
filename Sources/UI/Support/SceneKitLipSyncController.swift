import Foundation
import SceneKit

public struct SceneKitMorphMap: Sendable {
	public let mouthOpen: String
	public let mouthRound: String
	public let mouthNarrow: String
	
	public init(mouthOpen: String, mouthRound: String, mouthNarrow: String) {
		self.mouthOpen = mouthOpen
		self.mouthRound = mouthRound
		self.mouthNarrow = mouthNarrow
	}
}

/// A lightweight, text-driven lipsync controller for SceneKit avatars.
/// It approximates mouth shapes from streamed text deltas and animates morph targets.
/// For higher fidelity, prefer driving from audio and mapping to visemes.
@MainActor
public final class SceneKitLipSyncController {
	private weak var node: SCNNode?
	private let morphMap: SceneKitMorphMap
	
	private weak var morpher: SCNMorpher?
	private var targetNameToIndex: [String: Int] = [:]
	
	private var isActive: Bool = false
	private var timer: Timer?
	
	// Current weights we write into the morpher
	private var currentOpen: CGFloat = 0.0
	private var currentRound: CGFloat = 0.0
	private var currentNarrow: CGFloat = 0.0
	
	// How quickly weights decay toward zero per tick (60 Hz)
	private let decayPerTick: CGFloat = 0.08
	
	public init(node: SCNNode, morphMap: SceneKitMorphMap) {
		self.node = node
		self.morphMap = morphMap
		self.morpher = SceneKitLipSyncController.findMorpher(in: node)
		self.targetNameToIndex = SceneKitLipSyncController.buildTargetIndexMap(morpher: self.morpher)
	}
	
	public func startIfNeeded() {
		guard !isActive else { return }
		isActive = true
		startTimerIfNeeded()
	}
	
	public func handleDelta(_ delta: String) {
		guard !delta.isEmpty else { return }
		startIfNeeded()
		
		// Compute simple pulses from text content
		let lower = delta.lowercased()
		let vowels = CharacterSet(charactersIn: "aeiouyаеёиоуыэюя")
		let rounders = CharacterSet(charactersIn: "ouwоую")
		
		let length = lower.count
		let vowelCount = lower.unicodeScalars.filter { vowels.contains($0) }.count
		let roundCount = lower.unicodeScalars.filter { rounders.contains($0) }.count
		
		// Basic heuristics: more characters/vowels => larger open pulse
		let openPulse = clamp(0.2 + 0.02 * CGFloat(length) + 0.12 * CGFloat(vowelCount), 0.0, 1.0)
		let roundPulse = clamp(0.15 * CGFloat(roundCount), 0.0, 0.8)
		// Some consonants tighten the mouth slightly
		let consonantCount = max(0, length - vowelCount)
		let narrowPulse = clamp(0.05 * CGFloat(consonantCount), 0.0, 0.5)
		
		currentOpen = max(currentOpen, openPulse)
		currentRound = max(currentRound, roundPulse)
		currentNarrow = max(currentNarrow, narrowPulse)
		
		applyWeights()
	}
	
	public func stop(after delay: TimeInterval = 0.15) {
		guard isActive else { return }
		// Allow a short tail before fully stopping
		DispatchQueue.main.asyncAfter(deadline: .now() + delay) { [weak self] in
			guard let self else { return }
			self.currentOpen = 0
			self.currentRound = 0
			self.currentNarrow = 0
			self.applyWeights()
			self.stopTimerIfIdle()
			self.isActive = false
		}
	}
}

// MARK: - Private
@MainActor
private extension SceneKitLipSyncController {
	func startTimerIfNeeded() {
		guard timer == nil else { return }
		timer = Timer.scheduledTimer(withTimeInterval: 1.0 / 60.0, repeats: true) { [weak self] _ in
			self?.tick()
		}
		RunLoop.main.add(timer!, forMode: .common)
	}
	
	func stopTimerIfIdle() {
		guard currentOpen == 0, currentRound == 0, currentNarrow == 0 else { return }
		timer?.invalidate()
		timer = nil
	}
	
	func tick() {
		// Decay toward zero
		let newOpen = max(0, currentOpen - decayPerTick)
		let newRound = max(0, currentRound - decayPerTick)
		let newNarrow = max(0, currentNarrow - decayPerTick)
		
		// Early exit if nothing changes
		if newOpen == currentOpen, newRound == currentRound, newNarrow == currentNarrow {
			stopTimerIfIdle()
			return
		}
		
		currentOpen = newOpen
		currentRound = newRound
		currentNarrow = newNarrow
		applyWeights()
		
		stopTimerIfIdle()
	}
	
	func applyWeights() {
		guard let morpher else { return }
		
		if let openIdx = targetNameToIndex[morphMap.mouthOpen] {
			morpher.setWeight(currentOpen, forTargetAt: openIdx)
		}
		if let roundIdx = targetNameToIndex[morphMap.mouthRound] {
			morpher.setWeight(currentRound, forTargetAt: roundIdx)
		}
		if let narrowIdx = targetNameToIndex[morphMap.mouthNarrow] {
			morpher.setWeight(currentNarrow, forTargetAt: narrowIdx)
		}
	}
	
	static func findMorpher(in node: SCNNode) -> SCNMorpher? {
		if let morpher = node.geometry?.morpher {
			return morpher
		}
		for child in node.childNodes {
			if let m = findMorpher(in: child) {
				return m
			}
		}
		return nil
	}
	
	static func buildTargetIndexMap(morpher: SCNMorpher?) -> [String: Int] {
		guard let morpher, let targets = morpher.targets else { return [:] }
		var map: [String: Int] = [:]
		for (idx, target) in targets.enumerated() {
			if let name = target.name {
				map[name] = idx
			}
		}
		return map
	}
	
	func clamp(_ value: CGFloat, _ minVal: CGFloat, _ maxVal: CGFloat) -> CGFloat {
		return max(minVal, min(maxVal, value))
	}
}


