//
//  ImageQualityChecker.swift
//  FaceLivenessSDK
//
//  Created by Sreang on 22/3/25.
//

import Foundation
import UIKit
import MLKit

/**
 * Checks the quality of an image for face authentication
 */
@objc public class ImageQualityChecker: NSObject {
    private let TAG = "ImageQualityChecker"
    
    // Constants for brightness thresholds
    private let BRIGHTNESS_TOO_DARK: Float = 40.0
    private let BRIGHTNESS_SOMEWHAT_DARK: Float = 80.0
    private let BRIGHTNESS_GOOD_UPPER: Float = 180.0
    private let BRIGHTNESS_SOMEWHAT_BRIGHT: Float = 220.0
    
    private let SHARPNESS_VERY_BLURRY: Float = 5.0
    private let SHARPNESS_SOMEWHAT_BLURRY: Float = 10.0
    private let SHARPNESS_GOOD_UPPER: Float = 50.0
    private let SHARPNESS_TOO_DETAILED: Float = 100.0
    
    // Initialize face detector from ML Kit
    private lazy var detector: FaceDetector = {
        let options = FaceDetectorOptions()
        options.performanceMode = .accurate
        options.landmarkMode = .none
        options.classificationMode = .none
        return FaceDetector.faceDetector(options: options)
    }()
    
    /**
     * Check image quality metrics and face presence
     *
     * @param image The image to analyze
     * @return ImageQualityResult containing quality metrics
     * @throws QualityCheckException if quality check fails
     */
    @objc public func checkImageQuality(image: UIImage) throws -> ImageQualityResult {
        LogUtils.d(TAG, "Checking image quality for image: \(Int(image.size.width))x\(Int(image.size.height))")
        
        // Validate image
        guard BitmapUtils.validateImage(image) else {
            LogUtils.e(TAG, "Invalid image provided")
            throw QualityCheckException("Invalid image provided")
        }
        
        let result = ImageQualityResult()
        
        // Step 1: Check brightness
        result.brightnessScore = checkBrightness(image)
        
        // Step 2: Check sharpness
        result.sharpnessScore = checkSharpness(image)
        
        // Step 3: Check face presence
        do {
            let hasFace = try checkFacePresence(image)
            result.hasFace = hasFace
            result.faceScore = hasFace ? 1.0 : 0.0
            
            // Calculate overall score
            result.calculateOverallScore()
            
            return result
        } catch {
            LogUtils.e(TAG, "Error in quality check: \(error.localizedDescription)", error)
            throw QualityCheckException("Error in quality check: \(error.localizedDescription)", error)
        }
    }
    
    /**
     * Check image brightness - optimized with adaptive sampling
     *
     * @param image The image to analyze
     * @return Brightness score between 0.0 and 1.0
     */
    private func checkBrightness(_ image: UIImage) -> Float {
        guard let cgImage = image.cgImage else {
            return 0.5 // Default for invalid image
        }
        
        let w = cgImage.width
        let h = cgImage.height
        
        // Adaptive sampling - more pixels for smaller images, fewer for larger ones
        let stepSize = max(1, min(w, h) / 50)
        
        let avgBrightness = BitmapUtils.calculateAverageBrightness(image)
        
        // Convert brightness to a score using our defined constants
        let score: Float
        switch avgBrightness {
        case ..<BRIGHTNESS_TOO_DARK:
            // Too dark
            score = avgBrightness / BRIGHTNESS_TOO_DARK
            
        case BRIGHTNESS_TOO_DARK..<BRIGHTNESS_SOMEWHAT_DARK:
            // Somewhat dark
            score = 0.5 + (avgBrightness - BRIGHTNESS_TOO_DARK) / 80.0
            
        case BRIGHTNESS_SOMEWHAT_DARK..<BRIGHTNESS_GOOD_UPPER:
            // Good brightness
            score = 1.0
            
        case BRIGHTNESS_GOOD_UPPER..<BRIGHTNESS_SOMEWHAT_BRIGHT:
            // Somewhat bright
            score = 1.0 - (avgBrightness - BRIGHTNESS_GOOD_UPPER) / 80.0
            
        default:
            // Too bright
            score = 0.5 - (avgBrightness - BRIGHTNESS_SOMEWHAT_BRIGHT) / 70.0
        }
        
        // Ensure result is between 0 and 1
        return max(0.0, min(1.0, score))
    }
    
    /**
     * Check image sharpness by calculating gradients
     *
     * @param image The image to analyze
     * @return Sharpness score between 0.0 and 1.0
     */
    private func checkSharpness(_ image: UIImage) -> Float {
        // For iOS, we'll use a simpler method that uses the Laplacian operator to measure sharpness
        guard let cgImage = image.cgImage else {
            return 0.5 // Default for invalid image
        }
        
        let width = cgImage.width
        let height = cgImage.height
        
        // Skip processing for very small images
        if width < 10 || height < 10 {
            LogUtils.w(TAG, "Image too small for reliable sharpness calculation")
            return 0.5
        }
        
        let avgGrad = calculateLaplacianVariance(image)
        
        // Convert average gradient to a score using our constants
        let score: Float
        switch avgGrad {
        case ..<SHARPNESS_VERY_BLURRY:
            // Very blurry
            score = avgGrad / SHARPNESS_VERY_BLURRY
            
        case SHARPNESS_VERY_BLURRY..<SHARPNESS_SOMEWHAT_BLURRY:
            // Somewhat blurry
            score = 0.5 + (avgGrad - SHARPNESS_VERY_BLURRY) / 10.0
            
        case SHARPNESS_SOMEWHAT_BLURRY..<SHARPNESS_GOOD_UPPER:
            // Good sharpness
            score = 1.0
            
        case SHARPNESS_GOOD_UPPER..<SHARPNESS_TOO_DETAILED:
            // Too much detail/noise
            score = 1.0 - (avgGrad - SHARPNESS_GOOD_UPPER) / 100.0
            
        default:
            // Extremely noisy or artificially sharpened
            score = 0.5
        }
        
        // Ensure result is between 0 and 1
        return max(0.0, min(1.0, score))
    }
    
    /**
     * Calculate Laplacian variance - a measure of image sharpness
     */
    private func calculateLaplacianVariance(_ image: UIImage) -> Float {
        // This is a simplified implementation to estimate sharpness
        // We'll use the built-in Core Image filters to calculate edges
        guard let cgImage = image.cgImage else {
            return 0.0
        }
        
        let ciImage = CIImage(cgImage: cgImage)
        let context = CIContext(options: nil)
        
        // Apply an edge detection filter
        guard let edgeFilter = CIFilter(name: "CIEdges") else {
            return 0.0
        }
        
        edgeFilter.setValue(ciImage, forKey: kCIInputImageKey)
        
        guard let outputImage = edgeFilter.outputImage,
              let outputCGImage = context.createCGImage(outputImage, from: outputImage.extent) else {
            return 0.0
        }
        
        // Calculate the average intensity of the edge image
        let edgeUIImage = UIImage(cgImage: outputCGImage)
        let avgIntensity = BitmapUtils.calculateAverageBrightness(edgeUIImage)
        
        // Scale to appropriate range for sharpness measurement
        return avgIntensity * 0.5
    }
    
    /**
     * Check for face presence in the image using ML Kit
     */
    private func checkFacePresence(_ image: UIImage) throws -> Bool {
        let visionImage = VisionImage(image: image)
        visionImage.orientation = image.imageOrientation
        
        // Use a semaphore to make this synchronous
        let semaphore = DispatchSemaphore(value: 0)
        var detectedFaces: [Face] = []
        var detectionError: Error?
        
        detector.process(visionImage) { faces, error in
            if let error = error {
                detectionError = error
            } else if let faces = faces {
                detectedFaces = faces
            }
            semaphore.signal()
        }
        
        // Wait for the detection to complete
        _ = semaphore.wait(timeout: .now() + 5.0)
        
        // Check if there was an error
        if let error = detectionError {
            LogUtils.e(TAG, "Face detection failed: \(error.localizedDescription)", error)
            throw FaceDetectionException("Face detection failed: \(error.localizedDescription)", error)
        }
        
        let faceDetected = detectedFaces.count > 0
        LogUtils.d(TAG, "Face detection result: \(faceDetected ? "Face detected" : "No face detected")")
        
        return faceDetected
    }
    
    /**
     * Release resources when done
     */
    @objc public func close() {
        LogUtils.d(TAG, "ImageQualityChecker resources released")
    }
    
    deinit {
        close()
    }
}
