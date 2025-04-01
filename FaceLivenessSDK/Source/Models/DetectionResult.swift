//
//  DetectionResult.swift
//  FaceLivenessSDK
//
//  Created by Sreang on 22/3/25.
//

import Foundation

/**
 * Represents the result of a detection operation
 */
@objc public class DetectionResult: NSObject {
    /**
     * The prediction result label
     */
    @objc public let label: String
    
    /**
     * Confidence level in the prediction (0.0 to 1.0)
     */
    @objc public let confidence: Float
    
    /**
     * Initialize a new detection result
     */
    @objc public init(label: String, confidence: Float) {
        self.label = label
        self.confidence = confidence
        super.init()
    }
    
    public override var description: String {
        return "DetectionResult(label: \(label), confidence: \(confidence))"
    }
}
