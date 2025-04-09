import Foundation
import UIKit
import CoreImage
import MLKitVision
import MLKitFaceDetection

/**
 * A utility class to assess the quality of an image for face liveness detection.
 * Evaluates brightness, sharpness, and face presence using predefined thresholds.
 * Part of the FaceLivenessSDK for external use.
 */
@objc public class ImageQualityChecker: NSObject {
    private let TAG = "ImageQualityChecker"

    // Brightness thresholds
    private let BRIGHTNESS_TOO_DARK: Float = 40.0
    private let BRIGHTNESS_SOMEWHAT_DARK: Float = 80.0
    private let BRIGHTNESS_GOOD_UPPER: Float = 180.0
    private let BRIGHTNESS_SOMEWHAT_BRIGHT: Float = 220.0

    // Sharpness thresholds
    private let SHARPNESS_VERY_BLURRY: Float = 5.0
    private let SHARPNESS_SOMEWHAT_BLURRY: Float = 10.0
    private let SHARPNESS_GOOD_UPPER: Float = 50.0
    private let SHARPNESS_TOO_DETAILED: Float = 100.0

    private let faceDetector: FaceDetector

    /**
     * Initializes the ImageQualityChecker with default ML Kit face detection settings.
     * Suitable for external use in the FaceLivenessSDK.
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
     * Checks the quality of an image and returns the result asynchronously.
     *
     * @param image The UIImage to evaluate for brightness, sharpness, and face presence.
     * @param completion A closure called with the result. Provides either:
     *                   - result: An ImageQualityResult object if successful.
     *                   - error: An Error object if the check fails (nil if successful).
     */
    @objc public func checkImageQuality(image: UIImage, completion: @escaping (ImageQualityResult?, Error?) -> Void) {
        LogUtils.d(self.TAG, "Checking image quality for image: \(Int(image.size.width))x\(Int(image.size.height))")

        guard BitmapUtils.validateImage(image) else {
            LogUtils.e(self.TAG, "Invalid image provided")
            completion(nil, QualityCheckException("Invalid image provided"))
            return
        }

        let result = ImageQualityResult()

        result.brightnessScore = checkBrightness(image)
        result.sharpnessScore = checkSharpness(image)

        checkFacePresence(image) { hasFace, error in
            if let error = error {
                LogUtils.e(self.TAG, "Error in quality check: \(error.localizedDescription)", error)
                completion(nil, QualityCheckException("Error in quality check: \(error.localizedDescription)", error))
                return
            }

            result.hasFace = hasFace
            result.faceScore = hasFace ? 1.0 : 0.0
            result.calculateOverallScore()
            completion(result, nil)
        }
    }

    private func checkBrightness(_ image: UIImage) -> Float {
        let avgBrightness = BitmapUtils.calculateAverageBrightness(image)

        let score: Float
        switch avgBrightness {
        case ..<BRIGHTNESS_TOO_DARK: score = avgBrightness / BRIGHTNESS_TOO_DARK
        case BRIGHTNESS_TOO_DARK..<BRIGHTNESS_SOMEWHAT_DARK: score = 0.5 + (avgBrightness - BRIGHTNESS_TOO_DARK) / (BRIGHTNESS_SOMEWHAT_DARK - BRIGHTNESS_TOO_DARK)
        case BRIGHTNESS_SOMEWHAT_DARK..<BRIGHTNESS_GOOD_UPPER: score = 1.0
        case BRIGHTNESS_GOOD_UPPER..<BRIGHTNESS_SOMEWHAT_BRIGHT: score = 1.0 - (avgBrightness - BRIGHTNESS_GOOD_UPPER) / (BRIGHTNESS_SOMEWHAT_BRIGHT - BRIGHTNESS_GOOD_UPPER)
        default: score = 0.5 - (avgBrightness - BRIGHTNESS_SOMEWHAT_BRIGHT) / (255.0 - BRIGHTNESS_SOMEWHAT_BRIGHT)
        }

        return max(0.0, min(1.0, score))
    }

    private func checkSharpness(_ image: UIImage) -> Float {
        guard let cgImage = image.cgImage else { return 0.5 }

        let width = cgImage.width
        let height = cgImage.height

        if width < 10 || height < 10 {
            LogUtils.w(self.TAG, "Image too small for reliable sharpness calculation")
            return 0.5
        }

        let avgGrad = calculateLaplacianVariance(image)

        let score: Float
        switch avgGrad {
        case ..<SHARPNESS_VERY_BLURRY: score = avgGrad / SHARPNESS_VERY_BLURRY
        case SHARPNESS_VERY_BLURRY..<SHARPNESS_SOMEWHAT_BLURRY: score = 0.5 + (avgGrad - SHARPNESS_VERY_BLURRY) / (SHARPNESS_SOMEWHAT_BLURRY - SHARPNESS_VERY_BLURRY)
        case SHARPNESS_SOMEWHAT_BLURRY..<SHARPNESS_GOOD_UPPER: score = 1.0
        case SHARPNESS_GOOD_UPPER..<SHARPNESS_TOO_DETAILED: score = 1.0 - (avgGrad - SHARPNESS_GOOD_UPPER) / (SHARPNESS_TOO_DETAILED - SHARPNESS_GOOD_UPPER)
        default: score = 0.5
        }

        return max(0.0, min(1.0, score))
    }

    private func calculateLaplacianVariance(_ image: UIImage) -> Float {
        guard let cgImage = image.cgImage else { return 0.0 }

        let ciImage = CIImage(cgImage: cgImage)
        let context = CIContext(options: nil)

        guard let edgeFilter = CIFilter(name: "CIEdges") else { return 0.0 }

        edgeFilter.setValue(ciImage, forKey: kCIInputImageKey)

        guard let outputImage = edgeFilter.outputImage,
              let outputCGImage = context.createCGImage(outputImage, from: outputImage.extent) else {
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

    /**
     * Releases resources held by the ImageQualityChecker.
     * Call this when the checker is no longer needed to free up memory.
     */
    @objc public func close() {
        LogUtils.d(self.TAG, "ImageQualityChecker resources released")
    }

    deinit {
        close()
    }
}
