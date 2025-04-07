//ImageQualityChecker.swift
import Foundation
import UIKit
import CoreImage
import MLKitVision
import MLKitFaceDetection

@objc public class ImageQualityChecker: NSObject {
    private let TAG = "ImageQualityChecker"
    
    private let BRIGHTNESS_TOO_DARK: Float = 40.0
    private let BRIGHTNESS_SOMEWHAT_DARK: Float = 80.0
    private let BRIGHTNESS_GOOD_UPPER: Float = 180.0
    private let BRIGHTNESS_SOMEWHAT_BRIGHT: Float = 220.0
    
    private let SHARPNESS_VERY_BLURRY: Float = 5.0
    private let SHARPNESS_SOMEWHAT_BLURRY: Float = 10.0
    private let SHARPNESS_GOOD_UPPER: Float = 50.0
    private let SHARPNESS_TOO_DETAILED: Float = 100.0
    
    private let faceDetector: FaceDetector
    
    override init() {
        let options = FaceDetectorOptions()
        options.performanceMode = .fast
        options.minFaceSize = 0.2
        options.landmarkMode = .none
        options.classificationMode = .none
        self.faceDetector = FaceDetector.faceDetector(options: options)
        super.init()
    }
    
    @available(iOSApplicationExtension 13.0.0, *)
    @objc public func checkImageQuality(image: UIImage) async throws -> ImageQualityResult {
        LogUtils.d(TAG, "Checking image quality for image: \(Int(image.size.width))x\(Int(image.size.height))")
        
        guard BitmapUtils.validateImage(image) else {
            LogUtils.e(TAG, "Invalid image provided")
            throw QualityCheckException("Invalid image provided")
        }
        
        let result = ImageQualityResult()
        
        result.brightnessScore = checkBrightness(image)
        result.sharpnessScore = checkSharpness(image)
        
        do {
            let hasFace = try await checkFacePresence(image)
            result.hasFace = hasFace
            result.faceScore = hasFace ? 1.0 : 0.0
            result.calculateOverallScore()
            return result
        } catch {
            LogUtils.e(TAG, "Error in quality check: \(error.localizedDescription)", error)
            throw QualityCheckException("Error in quality check: \(error.localizedDescription)", error)
        }
    }
    
    private func checkBrightness(_ image: UIImage) -> Float {
        let avgBrightness = BitmapUtils.calculateAverageBrightness(image)
        
        let score: Float
        switch avgBrightness {
        case ..<BRIGHTNESS_TOO_DARK: score = avgBrightness / BRIGHTNESS_TOO_DARK
        case BRIGHTNESS_TOO_DARK..<BRIGHTNESS_SOMEWHAT_DARK: score = 0.5 + (avgBrightness - BRIGHTNESS_TOO_DARK) / 80.0
        case BRIGHTNESS_SOMEWHAT_DARK..<BRIGHTNESS_GOOD_UPPER: score = 1.0
        case BRIGHTNESS_GOOD_UPPER..<BRIGHTNESS_SOMEWHAT_BRIGHT: score = 1.0 - (avgBrightness - BRIGHTNESS_GOOD_UPPER) / 80.0
        default: score = 0.5 - (avgBrightness - BRIGHTNESS_SOMEWHAT_BRIGHT) / 70.0
        }
        
        return max(0.0, min(1.0, score))
    }
    
    private func checkSharpness(_ image: UIImage) -> Float {
        guard let cgImage = image.cgImage else { return 0.5 }
        
        let width = cgImage.width
        let height = cgImage.height
        
        if width < 10 || height < 10 {
            LogUtils.w(TAG, "Image too small for reliable sharpness calculation")
            return 0.5
        }
        
        let avgGrad = calculateLaplacianVariance(image)
        
        let score: Float
        switch avgGrad {
        case ..<SHARPNESS_VERY_BLURRY: score = avgGrad / SHARPNESS_VERY_BLURRY
        case SHARPNESS_VERY_BLURRY..<SHARPNESS_SOMEWHAT_BLURRY: score = 0.5 + (avgGrad - SHARPNESS_VERY_BLURRY) / 10.0
        case SHARPNESS_SOMEWHAT_BLURRY..<SHARPNESS_GOOD_UPPER: score = 1.0
        case SHARPNESS_GOOD_UPPER..<SHARPNESS_TOO_DETAILED: score = 1.0 - (avgGrad - SHARPNESS_GOOD_UPPER) / 100.0
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
    
    @available(iOSApplicationExtension 13.0.0, *)
    private func checkFacePresence(_ image: UIImage) async throws -> Bool {
        let visionImage = VisionImage(image: image)
        visionImage.orientation = image.imageOrientation
        do {
            let faces = try await faceDetector.process(visionImage)
            let faceDetected = !faces.isEmpty
            LogUtils.d(TAG, "Face detection result: \(faceDetected ? "Face detected" : "No face detected")")
            return faceDetected
        } catch {
            LogUtils.e(TAG, "MLKit face detection failed: \(error.localizedDescription)")
            throw FaceDetectionException("MLKit face detection failed: \(error.localizedDescription)", error)
        }
    }
    
    @objc public func close() {
        LogUtils.d(TAG, "ImageQualityChecker resources released")
    }
    
    deinit {
        close()
    }
}


