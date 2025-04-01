//
//  BitmapUtils.swift
//  FaceLivenessSDK
//
//  Created by Sreang on 22/3/25.
//

import Foundation
import UIKit

/**
 * Utility functions for image processing and manipulation
 */
@objc public class BitmapUtils: NSObject {
    private static let TAG = "BitmapUtils"
    
    /**
     * Constants for validation
     */
    @objc public static let MIN_IMAGE_SIZE = 64 // Minimum size in pixels
    @objc public static let MAX_IMAGE_SIZE = 4096 // Maximum size in pixels
    
    /**
     * Validates the input image
     *
     * @param image Image to validate
     * @return true if valid, false otherwise
     */
    @objc public static func validateImage(_ image: UIImage?) -> Bool {
        guard let image = image else {
            LogUtils.e(TAG, "Input image is null")
            return false
        }
        
        let width = Int(image.size.width * image.scale)
        let height = Int(image.size.height * image.scale)
        
        if width <= MIN_IMAGE_SIZE || height <= MIN_IMAGE_SIZE {
            LogUtils.e(TAG, "Image too small: \(width)x\(height)")
            return false
        }
        
        if width >= MAX_IMAGE_SIZE || height >= MAX_IMAGE_SIZE {
            LogUtils.e(TAG, "Image too large: \(width)x\(height)")
            return false
        }
        
        if image.cgImage == nil {
            LogUtils.e(TAG, "Image has no CGImage representation")
            return false
        }
        
        return true
    }
    
    /**
     * Resizes an image to the specified dimensions
     *
     * @param image Source image
     * @param width Target width
     * @param height Target height
     * @return Resized image
     */
    @objc public static func resizeImage(_ image: UIImage, width: Int, height: Int) -> UIImage? {
        let currentWidth = Int(image.size.width * image.scale)
        let currentHeight = Int(image.size.height * image.scale)
        
        if currentWidth == width && currentHeight == height {
            return image
        }
        
        UIGraphicsBeginImageContextWithOptions(CGSize(width: width, height: height), false, 0.0)
        defer { UIGraphicsEndImageContext() }
        
        image.draw(in: CGRect(x: 0, y: 0, width: width, height: height))
        
        guard let resizedImage = UIGraphicsGetImageFromCurrentImageContext() else {
            LogUtils.e(TAG, "Failed to resize image")
            return nil
        }
        
        return resizedImage
    }
    
    /**
     * Calculate average brightness of an image
     *
     * @param image Image to analyze
     * @return Average brightness value (0-255)
     */
    @objc public static func calculateAverageBrightness(_ image: UIImage) -> Float {
        guard let cgImage = image.cgImage else {
            LogUtils.e(TAG, "Cannot calculate brightness: no CGImage")
            return 0.0
        }
        
        let width = cgImage.width
        let height = cgImage.height
        
        // Adaptive sampling for better performance
        let stepSize = max(1, min(width, height) / 50)
        
        guard let context = createARGBBitmapContext(from: cgImage) else {
            LogUtils.e(TAG, "Failed to create bitmap context")
            return 0.0
        }
        
        guard let data = context.data else {
            LogUtils.e(TAG, "No bitmap data available")
            return 0.0
        }
        
        let pixelData = data.bindMemory(to: UInt8.self, capacity: width * height * 4)
        var total: Int64 = 0
        var count = 0
        
        for y in stride(from: 0, to: height, by: stepSize) {
            for x in stride(from: 0, to: width, by: stepSize) {
                let offset = 4 * (y * width + x)
                let r = Int(pixelData[offset])
                let g = Int(pixelData[offset + 1])
                let b = Int(pixelData[offset + 2])
                
                // Calculate perceived brightness using standard luminance formula
                let brightness = Int((0.299 * Float(r) + 0.587 * Float(g) + 0.114 * Float(b)))
                total += Int64(brightness)
                count += 1
            }
        }
        
        return Float(total) / Float(count)
    }
    
    /**
     * Creates a bitmap context for pixel access
     */
    private static func createARGBBitmapContext(from image: CGImage) -> CGContext? {
        let width = image.width
        let height = image.height
        let bytesPerRow = width * 4
        let bitmapInfo = CGImageAlphaInfo.premultipliedFirst.rawValue | CGBitmapInfo.byteOrder32Little.rawValue
        
        guard let context = CGContext(data: nil,
                                     width: width,
                                     height: height,
                                     bitsPerComponent: 8,
                                     bytesPerRow: bytesPerRow,
                                     space: CGColorSpaceCreateDeviceRGB(),
                                     bitmapInfo: bitmapInfo) else {
            return nil
        }
        
        context.draw(image, in: CGRect(x: 0, y: 0, width: width, height: height))
        return context
    }
    
    /**
     * Convert UIImage to normalized float array for ML model input
     *
     * @param image The image to convert
     * @param width Target width
     * @param height Target height
     * @param means The normalization means for RGB channels
     * @param stds The normalization standard deviations for RGB channels
     * @return Float array of normalized pixel values
     */
    @objc public static func normalizeImage(_ image: UIImage, width: Int, height: Int, means: [Float], stds: [Float]) -> [Float]? {
        guard let resizedImage = resizeImage(image, width: width, height: height),
              let cgImage = resizedImage.cgImage else {
            return nil
        }
        
        guard let context = createARGBBitmapContext(from: cgImage) else {
            return nil
        }
        
        guard let data = context.data else {
            return nil
        }
        
        let bytesPerRow = cgImage.bytesPerRow
        let pixelData = data.bindMemory(to: UInt8.self, capacity: height * bytesPerRow)
        
        // Create the destination array (NCHW format: batch, channels, height, width)
        var normalizedData = [Float](repeating: 0.0, count: 3 * height * width)
        
        // Process each channel separately (R, G, B)
        for c in 0..<3 {
            let channelOffset = c * height * width
            let meanVal = means[c]
            let stdVal = stds[c]
            
            for h in 0..<height {
                for w in 0..<width {
                    let pixelOffset = h * bytesPerRow + w * 4
                    
                    // Get the appropriate channel value (BGRA format in memory)
                    let channelIndex = [2, 1, 0][c] // R=2, G=1, B=0 in BGRA
                    let pixelValue = Float(pixelData[pixelOffset + channelIndex]) / 255.0
                    
                    // Apply normalization
                    normalizedData[channelOffset + h * width + w] = (pixelValue - meanVal) / stdVal
                }
            }
        }
        
        return normalizedData
    }
}
