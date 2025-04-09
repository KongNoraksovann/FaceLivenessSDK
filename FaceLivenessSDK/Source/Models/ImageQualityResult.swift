import Foundation

/**
 * Represents the result of an image quality check in the FaceLivenessSDK.
 * Includes scores for brightness, sharpness, and face presence, with an overall weighted score.
 */
@objc public class ImageQualityResult: NSObject {
    @objc public var brightnessScore: Float = 0.0
    @objc public var sharpnessScore: Float = 0.0
    @objc public var faceScore: Float = 0.0
    @objc public var hasFace: Bool = false
    @objc public var overallScore: Float = 0.0
    
    @objc public static let brightnessWeight: Float = 0.3
    @objc public static let sharpnessWeight: Float = 0.3
    @objc public static let faceWeight: Float = 0.4
    @objc public static let acceptableScoreThreshold: Float = 0.5
    
    @objc public static func createDefault() -> ImageQualityResult {
        let result = ImageQualityResult()
        result.brightnessScore = 0.0
        result.sharpnessScore = 0.0
        result.faceScore = 0.0
        result.hasFace = false
        result.overallScore = 0.0
        return result
    }
    
    @objc public func calculateOverallScore() {
        if !hasFace {
            overallScore = 0.0
        } else {
            overallScore = (brightnessScore * Self.brightnessWeight +
                            sharpnessScore * Self.sharpnessWeight +
                            faceScore * Self.faceWeight)
            overallScore = max(0.0, min(1.0, overallScore))
        }
    }
    
    @objc public func isAcceptable() -> Bool {
        return hasFace && overallScore >= Self.acceptableScoreThreshold
    }
    
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
    
    @objc public override var description: String {
        return String(format: "Quality: %.2f (Brightness: %.2f, Sharpness: %.2f, Face: %.2f)",
                      overallScore, brightnessScore, sharpnessScore, faceScore)
    }
}
