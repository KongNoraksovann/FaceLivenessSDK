//
//  ImageQualityChecker.swift
//  FaceLivenessSDK
//
//  Created by Sovann on 22/3/25.
//

import Foundation
import UIKit
import CoreImage

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
        
        // Step 3: Check face presence (using simple heuristic instead of MLKit)
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
    
    // Brightness check remains the same
    private func checkBrightness(_ image: UIImage) -> Float {
        guard let cgImage = image.cgImage else {
            return 0.5 // Default for invalid image
        }
        
        let w = cgImage.width
        let h = cgImage.height
        
        let stepSize = max(1, min(w, h) / 50)
        
        let avgBrightness = BitmapUtils.calculateAverageBrightness(image)
        
        let score: Float
        switch avgBrightness {
        case ..<BRIGHTNESS_TOO_DARK:
            score = avgBrightness / BRIGHTNESS_TOO_DARK
        case BRIGHTNESS_TOO_DARK..<BRIGHTNESS_SOMEWHAT_DARK:
            score = 0.5 + (avgBrightness - BRIGHTNESS_TOO_DARK) / 80.0
        case BRIGHTNESS_SOMEWHAT_DARK..<BRIGHTNESS_GOOD_UPPER:
            score = 1.0
        case BRIGHTNESS_GOOD_UPPER..<BRIGHTNESS_SOMEWHAT_BRIGHT:
            score = 1.0 - (avgBrightness - BRIGHTNESS_GOOD_UPPER) / 80.0
        default:
            score = 0.5 - (avgBrightness - BRIGHTNESS_SOMEWHAT_BRIGHT) / 70.0
        }
        
        return max(0.0, min(1.0, score))
    }
    
    // Sharpness check remains the same
    private func checkSharpness(_ image: UIImage) -> Float {
        guard let cgImage = image.cgImage else {
            return 0.5
        }
        
        let width = cgImage.width
        let height = cgImage.height
        
        if width < 10 || height < 10 {
            LogUtils.w(TAG, "Image too small for reliable sharpness calculation")
            return 0.5
        }
        
        let avgGrad = calculateLaplacianVariance(image)
        
        let score: Float
        switch avgGrad {
        case ..<SHARPNESS_VERY_BLURRY:
            score = avgGrad / SHARPNESS_VERY_BLURRY
        case SHARPNESS_VERY_BLURRY..<SHARPNESS_SOMEWHAT_BLURRY:
            score = 0.5 + (avgGrad - SHARPNESS_VERY_BLURRY) / 10.0
        case SHARPNESS_SOMEWHAT_BLURRY..<SHARPNESS_GOOD_UPPER:
            score = 1.0
        case SHARPNESS_GOOD_UPPER..<SHARPNESS_TOO_DETAILED:
            score = 1.0 - (avgGrad - SHARPNESS_GOOD_UPPER) / 100.0
        default:
            score = 0.5
        }
        
        return max(0.0, min(1.0, score))
    }
    
    private func calculateLaplacianVariance(_ image: UIImage) -> Float {
        guard let cgImage = image.cgImage else {
            return 0.0
        }
        
        let ciImage = CIImage(cgImage: cgImage)
        let context = CIContext(options: nil)
        
        guard let edgeFilter = CIFilter(name: "CIEdges") else {
            return 0.0
        }
        
        edgeFilter.setValue(ciImage, forKey: kCIInputImageKey)
        
        guard let outputImage = edgeFilter.outputImage,
              let outputCGImage = context.createCGImage(outputImage, from: outputImage.extent) else {
            return 0.0
        }
        
        let edgeUIImage = UIImage(cgImage: outputCGImage)
        let avgIntensity = BitmapUtils.calculateAverageBrightness(edgeUIImage)
        
        return avgIntensity * 0.5
    }
    
    /**
     * Simple face presence check using CoreImage instead of MLKit
     * Note: This is a basic implementation and less accurate than MLKit
     */
    private func checkFacePresence(_ image: UIImage) throws -> Bool {
        let ciImage = CIImage(image: image)
        guard let detector = CIDetector(
            ofType: CIDetectorTypeFace,
            context: nil,
            options: [CIDetectorAccuracy: CIDetectorAccuracyHigh]
        ) else {
            throw FaceDetectionException("Could not create face detector")
        }
        
        let features = detector.features(in: ciImage!)
        let faceDetected = !features.isEmpty
        
        LogUtils.d(TAG, "Face detection result: \(faceDetected ? "Face detected" : "No face detected")")
        return faceDetected
    }
    
    @objc public func close() {
        LogUtils.d(TAG, "ImageQualityChecker resources released")
    }
    
    deinit {
        close()
    }
}
