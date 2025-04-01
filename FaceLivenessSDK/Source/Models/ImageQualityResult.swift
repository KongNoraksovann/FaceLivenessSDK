//
//  ImageQualityResult.swift
//  FaceLivenessSDK
//
//  Created by Sreang on 22/3/25.
//

import Foundation

/**
 * Represents the result of image quality check
 */
@objc public class ImageQualityResult: NSObject {
    @objc public var brightnessScore: Float = 0.0
    @objc public var sharpnessScore: Float = 0.0
    @objc public var faceScore: Float = 0.0
    @objc public var hasFace: Bool = false
    @objc public var overallScore: Float = 0.0
    
    // Weights for each component - made as constants for better maintainability
    @objc public static let BRIGHTNESS_WEIGHT: Float = 0.3
    @objc public static let SHARPNESS_WEIGHT: Float = 0.3
    @objc public static let FACE_WEIGHT: Float = 0.4
    
    // Minimum acceptable overall score
    @objc public static let ACCEPTABLE_SCORE_THRESHOLD: Float = 0.5
    
    /**
     * Create a default instance for cases where quality check is skipped
     */
    @objc public static func createDefault() -> ImageQualityResult {
        let result = ImageQualityResult()
        result.brightnessScore = 0.0
        result.sharpnessScore = 0.0
        result.faceScore = 0.0
        result.hasFace = false  // Important: this will cause isAcceptable() to return false
        result.overallScore = 0.0
        return result
    }
    
    /**
     * Calculates the overall score based on weighted components
     */
    @objc public func calculateOverallScore() {
        if !hasFace {
            overallScore = 0.0
        } else {
            overallScore = (brightnessScore * ImageQualityResult.BRIGHTNESS_WEIGHT +
                    sharpnessScore * ImageQualityResult.SHARPNESS_WEIGHT +
                    faceScore * ImageQualityResult.FACE_WEIGHT)
            
            // Ensure score is between 0 and 1
            overallScore = max(0.0, min(1.0, overallScore))
        }
    }
    
    /**
     * Determines if the image quality is acceptable for further processing
     */
    @objc public func isAcceptable() -> Bool {
        return hasFace && overallScore >= ImageQualityResult.ACCEPTABLE_SCORE_THRESHOLD
    }
    
    /**
     * Get detailed breakdown of all component scores
     */
    @objc public func getDetailedReport() -> [String: Any] {
        return [
            "overallScore": overallScore,
            "brightnessScore": brightnessScore,
            "sharpnessScore": sharpnessScore,
            "faceScore": faceScore,
            "hasFace": hasFace,
            "isAcceptable": isAcceptable()
        ]
    }
    
    public override var description: String {
        return String(format: "Quality: %.2f (Brightness: %.2f, Sharpness: %.2f, Face: %.2f)",
                     overallScore, brightnessScore, sharpnessScore, faceScore)
    }
}
