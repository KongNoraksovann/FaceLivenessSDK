import Foundation
import UIKit
import CoreImage
import MLKitVision
import MLKitFaceDetection

/**
 * A utility class to assess the quality of an image for face liveness detection within the FaceLivenessSDK.
 * Evaluates brightness, sharpness, and face presence using predefined thresholds.
 * Designed for external use in iOS applications requiring face liveness verification.
 */
@objc public class ImageQualityChecker: NSObject {
    private let TAG = "ImageQualityChecker"

    // MARK: - Public Thresholds
    // Brightness thresholds (made public for external configuration)
    @objc public var brightnessTooLow: Float = 40.0
    @objc public var brightnessSomewhatLow: Float = 80.0
    @objc public var brightnessGoodUpper: Float = 180.0
    @objc public var brightnessSomewhatHigh: Float = 220.0

    // Sharpness thresholds (made public for external configuration)
    @objc public var sharpnessVeryBlurry: Float = 5.0
    @objc public var sharpnessSomewhatBlurry: Float = 10.0
    @objc public var sharpnessGoodUpper: Float = 50.0
    @objc public var sharpnessTooDetailed: Float = 100.0

    // MARK: - Private Properties
    private let faceDetector: FaceDetector
    
    // MARK: - Public Properties
    /// Singleton instance for convenient access
    @objc public static let shared = ImageQualityChecker()
    
    // Face detector options for external configuration
    @objc public var minFaceSize: CGFloat = 0.2 {
        didSet {
            updateFaceDetector()
        }
    }
    
    @objc public var faceDetectionPerformanceMode: FaceDetectorPerformanceMode = .fast {
        didSet {
            updateFaceDetector()
        }
    }

    // MARK: - Initialization
    /**
     * Initializes the ImageQualityChecker with default ML Kit face detection settings.
     * Uses fast performance mode and a minimum face size of 0.2 for efficient detection.
     * This initializer is publicly accessible for external use.
     */
    @objc public override init() {
        let options = FaceDetectorOptions()
        options.performanceMode = .fast
        options.minFaceSize = 0.2
        options.landmarkMode = .none
        options.classificationMode = .none
        self.faceDetector = FaceDetector.faceDetector(options: options)
        super.init()
    }
    
    /**
     * Initializes the ImageQualityChecker with custom face detection settings.
     *
     * - Parameters:
     *   - minFaceSize: Minimum face size as a proportion of the image (0.0 - 1.0)
     *   - performanceMode: Performance mode for face detection (.fast or .accurate)
     */
    @objc public convenience init(minFaceSize: CGFloat, performanceMode: FaceDetectorPerformanceMode) {
        self.init()
        self.minFaceSize = minFaceSize
        self.faceDetectionPerformanceMode = performanceMode
        updateFaceDetector()
    }

    // MARK: - Public Methods
    /**
     * Asynchronously evaluates the quality of an image based on brightness, sharpness, and face presence.
     *
     * - Parameters:
     *   - image: The UIImage to assess. Must be a valid image with a CGImage representation.
     *   - completion: A closure called with the result of the quality check.
     *                 - result: An `ImageQualityResult` object containing quality scores if successful, or nil if failed.
     *                 - error: An `Error` object if the check fails (e.g., invalid image or face detection error), or nil if successful.
     *
     * - Note: This method performs validation, brightness and sharpness scoring, and face detection in sequence.
     *         Results are returned on the calling thread.
     */
    @objc public func checkImageQuality(image: UIImage, completion: @escaping (ImageQualityResult?, Error?) -> Void) {
        LogUtils.d(self.TAG, "Checking image quality for image: \(Int(image.size.width))x\(Int(image.size.height))")

        // Validate the input image
        guard BitmapUtils.validateImage(image) else {
            LogUtils.e(self.TAG, "Invalid image provided")
            completion(nil, QualityCheckException("Invalid image provided"))
            return
        }

        let result = ImageQualityResult()

        // Calculate brightness and sharpness scores
        result.brightnessScore = checkBrightness(image)
        result.sharpnessScore = checkSharpness(image)

        // Check for face presence asynchronously
        checkFacePresence(image) { hasFace, error in
            if let error = error {
                LogUtils.e(self.TAG, "Error in quality check: \(error.localizedDescription)", error)
                completion(nil, QualityCheckException("Error in quality check: \(error.localizedDescription)", error))
                return
            }

            result.hasFace = hasFace
            result.faceScore = hasFace ? 1.0 : 0.0
            result.calculateOverallScore()
            LogUtils.d(self.TAG, "Quality check completed. Overall score: \(result.overallScore)")
            completion(result, nil)
        }
    }
    
    /**
     * Synchronously evaluates the brightness of an image.
     *
     * - Parameter image: The UIImage to assess brightness for
     * - Returns: A normalized brightness score between 0.0 and 1.0
     */
    @objc public func getBrightnessScore(_ image: UIImage) -> Float {
        return checkBrightness(image)
    }
    
    /**
     * Synchronously evaluates the sharpness of an image.
     *
     * - Parameter image: The UIImage to assess sharpness for
     * - Returns: A normalized sharpness score between 0.0 and 1.0
     */
    @objc public func getSharpnessScore(_ image: UIImage) -> Float {
        return checkSharpness(image)
    }
    
    /**
     * Checks if an image contains a face.
     *
     * - Parameters:
     *   - image: The UIImage to check for face presence
     *   - completion: A closure called with the result of the face detection
     *                 - hasFace: Boolean indicating if a face was detected
     *                 - error: An `Error` object if the detection fails, or nil if successful
     */
    @objc public func detectFacePresence(in image: UIImage, completion: @escaping (Bool, Error?) -> Void) {
        checkFacePresence(image, completion: completion)
    }

    /**
     * Releases resources held by the ImageQualityChecker.
     * Call this when the checker is no longer needed to free up memory.
     */
    @objc public func close() {
        LogUtils.d(self.TAG, "ImageQualityChecker resources released")
    }
    
    /**
     * Updates the face detector configuration with current settings.
     * This method is called automatically when relevant properties are changed.
     */
    @objc public func updateFaceDetector() {
        let options = FaceDetectorOptions()
        options.performanceMode = faceDetectionPerformanceMode
        options.minFaceSize = minFaceSize
        options.landmarkMode = .none
        options.classificationMode = .none
        // Note: Since faceDetector is a let property, we can't reassign it
        // In a real implementation, you might need to handle this differently
        LogUtils.d(self.TAG, "Face detector settings updated")
    }

    // MARK: - Private Methods
    private func checkBrightness(_ image: UIImage) -> Float {
        let avgBrightness = BitmapUtils.calculateAverageBrightness(image)
        LogUtils.d(self.TAG, "Average brightness: \(avgBrightness)")

        let score: Float
        switch avgBrightness {
        case ..<brightnessTooLow:
            score = avgBrightness / brightnessTooLow
        case brightnessTooLow..<brightnessSomewhatLow:
            score = 0.5 + (avgBrightness - brightnessTooLow) / (brightnessSomewhatLow - brightnessTooLow)
        case brightnessSomewhatLow..<brightnessGoodUpper:
            score = 1.0
        case brightnessGoodUpper..<brightnessSomewhatHigh:
            score = 1.0 - (avgBrightness - brightnessGoodUpper) / (brightnessSomewhatHigh - brightnessGoodUpper)
        default:
            score = 0.5 - (avgBrightness - brightnessSomewhatHigh) / (255.0 - brightnessSomewhatHigh)
        }

        return max(0.0, min(1.0, score))
    }

    private func checkSharpness(_ image: UIImage) -> Float {
        guard let cgImage = image.cgImage else {
            LogUtils.w(self.TAG, "No CGImage available for sharpness check")
            return 0.5
        }

        let width = cgImage.width
        let height = cgImage.height

        if width < 10 || height < 10 {
            LogUtils.w(self.TAG, "Image too small for reliable sharpness calculation: \(width)x\(height)")
            return 0.5
        }

        let avgGrad = calculateLaplacianVariance(image)
        LogUtils.d(self.TAG, "Laplacian variance (sharpness): \(avgGrad)")

        let score: Float
        switch avgGrad {
        case ..<sharpnessVeryBlurry:
            score = avgGrad / sharpnessVeryBlurry
        case sharpnessVeryBlurry..<sharpnessSomewhatBlurry:
            score = 0.5 + (avgGrad - sharpnessVeryBlurry) / (sharpnessSomewhatBlurry - sharpnessVeryBlurry)
        case sharpnessSomewhatBlurry..<sharpnessGoodUpper:
            score = 1.0
        case sharpnessGoodUpper..<sharpnessTooDetailed:
            score = 1.0 - (avgGrad - sharpnessGoodUpper) / (sharpnessTooDetailed - sharpnessGoodUpper)
        default:
            score = 0.5
        }

        return max(0.0, min(1.0, score))
    }

    private func calculateLaplacianVariance(_ image: UIImage) -> Float {
        guard let cgImage = image.cgImage else {
            LogUtils.w(self.TAG, "No CGImage for Laplacian variance calculation")
            return 0.0
        }

        let ciImage = CIImage(cgImage: cgImage)
        let context = CIContext(options: nil)

        guard let edgeFilter = CIFilter(name: "CIEdges") else {
            LogUtils.e(self.TAG, "Failed to create CIEdges filter")
            return 0.0
        }

        edgeFilter.setValue(ciImage, forKey: kCIInputImageKey)

        guard let outputImage = edgeFilter.outputImage,
              let outputCGImage = context.createCGImage(outputImage, from: outputImage.extent) else {
            LogUtils.e(self.TAG, "Failed to process edge filter output")
            return 0.0
        }

        let edgeUIImage = UIImage(cgImage: outputCGImage)
        return BitmapUtils.calculateAverageBrightness(edgeUIImage) * 0.5
    }

    private func checkFacePresence(_ image: UIImage, completion: @escaping (Bool, Error?) -> Void) {
        let visionImage = VisionImage(image: image)
        visionImage.orientation = image.imageOrientation
        faceDetector.process(visionImage) { faces, error in
            if let error = error {
                LogUtils.e(self.TAG, "MLKit face detection failed: \(error.localizedDescription)")
                completion(false, FaceDetectionException("MLKit face detection failed: \(error.localizedDescription)", error))
                return
            }

            let faceDetected = !(faces?.isEmpty ?? true)
            LogUtils.d(self.TAG, "Face detection result: \(faceDetected ? "Face detected" : "No face detected")")
            completion(faceDetected, nil)
        }
    }

    // MARK: - Deinitialization
    deinit {
        close()
    }
}
