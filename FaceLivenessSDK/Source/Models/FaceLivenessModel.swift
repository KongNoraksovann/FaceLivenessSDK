//
//  FaceLivenessModel.swift
//  FaceLivenessSDK
//
//  Created by Sreang on 22/3/25.
//

import Foundation

/**
 * Represents the result of the face liveness detection process
 */
@objc public class FaceLivenessModel: NSObject {
    /**
     * The prediction result: "Live" or "Spoof"
     */
    @objc public let prediction: String
    
    /**
     * Confidence level in the prediction (0.0 to 1.0)
     */
    @objc public let confidence: Float
    
    /**
     * Reason for failure if authentication failed
     */
    @objc public let failureReason: String?
    
    /**
     * Initialize a new face liveness model
     */
    @objc public init(prediction: String, confidence: Float, failureReason: String? = nil) {
        self.prediction = prediction
        self.confidence = confidence
        self.failureReason = failureReason
        super.init()
    }
    
    /**
     * Check if the liveness check passed successfully
     */
    @objc public var isLive: Bool {
        return prediction == "Live"
    }
    
    public override var description: String {
        if let reason = failureReason {
            return "FaceLivenessModel(prediction: \(prediction), confidence: \(confidence)), reason: \(reason))"
        } else {
            return "FaceLivenessModel(prediction: \(prediction), confidence: \(confidence)))"
        }
    }
}
