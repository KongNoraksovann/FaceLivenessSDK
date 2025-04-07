// BitmapUtils.swift
import Foundation
import UIKit
import onnxruntime_objc

/**
 * Utility functions for image processing and manipulation
 */
@objc public class BitmapUtils: NSObject {
    private static let TAG = "BitmapUtils"
    
    // ImageNet normalization values
    private static let mean: [Float] = [0.485, 0.456, 0.406]
    private static let std: [Float] = [0.229, 0.224, 0.225]
    
    @objc public static let MIN_IMAGE_SIZE = 64
    @objc public static let MAX_IMAGE_SIZE = 4096
    
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
    
    @objc public static func resizeImage(_ image: UIImage, width: Int, height: Int) -> UIImage? {
        let size = CGSize(width: width, height: height)
        UIGraphicsBeginImageContextWithOptions(size, false, 1.0)
        defer { UIGraphicsEndImageContext() }
        
        image.draw(in: CGRect(origin: .zero, size: size))
        return UIGraphicsGetImageFromCurrentImageContext()
    }
    
    @objc public static func calculateAverageBrightness(_ image: UIImage) -> Float {
        guard let cgImage = image.cgImage else {
            LogUtils.e(TAG, "Cannot calculate brightness: no CGImage")
            return 0.0
        }
        
        let width = cgImage.width
        let height = cgImage.height
        
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
                
                let brightness = Int((0.299 * Float(r) + 0.587 * Float(g) + 0.114 * Float(b)))
                total += Int64(brightness)
                count += 1
            }
        }
        
        return Float(total) / Float(count)
    }
    
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
        
        var normalizedData = [Float](repeating: 0.0, count: 3 * height * width)
        
        for c in 0..<3 {
            let channelOffset = c * height * width
            let meanVal = means[c]
            let stdVal = stds[c]
            
            for h in 0..<height {
                for w in 0..<width {
                    let pixelOffset = h * bytesPerRow + w * 4
                    let channelIndex = [2, 1, 0][c]
                    let pixelValue = Float(pixelData[pixelOffset + channelIndex]) / 255.0
                    normalizedData[channelOffset + h * width + w] = (pixelValue - meanVal) / stdVal
                }
            }
        }
        
        return normalizedData
    }
    
    @objc public static func preprocessImage(_ image: UIImage, size: Int) -> ORTValue? {
        let inputWidth = size
        let inputHeight = size
        let pixelCount = inputWidth * inputHeight
        
        guard validateImage(image) else {
            LogUtils.e(TAG, "Invalid input image")
            return nil
        }
        
        guard let resizedImage = resizeImage(image, width: inputWidth, height: inputHeight) else {
            LogUtils.e(TAG, "Failed to resize image")
            return nil
        }
        
        guard let cgImage = resizedImage.cgImage else {
            LogUtils.e(TAG, "Failed to get CGImage")
            return nil
        }
        
        let width = cgImage.width
        let height = cgImage.height
        let byteCount = pixelCount * 4
        
        LogUtils.d(TAG, "Image dimensions after resize - width: \(width), height: \(height), pixelCount: \(pixelCount)")
        
        guard width == inputWidth, height == inputHeight else {
            LogUtils.e(TAG, "Resized image dimensions (\(width)x\(height)) do not match expected (\(inputWidth)x\(inputHeight))")
            return nil
        }
        
        let data = UnsafeMutableBufferPointer<UInt8>.allocate(capacity: byteCount)
        defer { data.deallocate() }
        data.initialize(repeating: 0)
        
        let colorSpace = CGColorSpaceCreateDeviceRGB()
        guard let context = CGContext(
            data: data.baseAddress,
            width: width,
            height: height,
            bitsPerComponent: 8,
            bytesPerRow: width * 4,
            space: colorSpace,
            bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue | CGBitmapInfo.byteOrder32Big.rawValue
        ) else {
            LogUtils.e(TAG, "Failed to create CGContext")
            return nil
        }
        
        context.draw(cgImage, in: CGRect(x: 0, y: 0, width: width, height: height))
        
        let middlePixelIdx = pixelCount / 2
        let r = Float(data[middlePixelIdx * 4]) / 255.0
        let g = Float(data[middlePixelIdx * 4 + 1]) / 255.0
        let b = Float(data[middlePixelIdx * 4 + 2]) / 255.0
        LogUtils.d(TAG, "Sample middle pixel RGB normalized: [\(r), \(g), \(b)]")
        LogUtils.d(TAG, "Sample middle pixel after normalization: [{\((r - mean[0]) / std[0])}, \((g - mean[1]) / std[1]), \((b - mean[2]) / std[2])]")
        
        let expectedSize = 1 * 3 * inputWidth * inputHeight
        var floatArray = [Float](repeating: 0.0, count: expectedSize)
        LogUtils.d(TAG, "floatArray size: \(floatArray.count), expected: \(expectedSize)")
        
        for i in 0..<pixelCount {
            let pixelOffset = i * 4
            let r = Float(data[pixelOffset]) / 255.0
            floatArray[i] = (r - mean[0]) / std[0]
        }
        
        for i in 0..<pixelCount {
            let pixelOffset = i * 4 + 1
            let g = Float(data[pixelOffset]) / 255.0
            floatArray[i + pixelCount] = (g - mean[1]) / std[1]
        }
        
        for i in 0..<pixelCount {
            let pixelOffset = i * 4 + 2
            let b = Float(data[pixelOffset]) / 255.0
            floatArray[i + 2 * pixelCount] = (b - mean[2]) / std[2]
        }
        
        do {
            let shape: [NSNumber] = [1, 3, NSNumber(value: inputHeight), NSNumber(value: inputWidth)]
            let tensorData = NSMutableData(bytes: floatArray, length: floatArray.count * MemoryLayout<Float>.size)
            return try ORTValue(tensorData: tensorData, elementType: .float, shape: shape)
        } catch {
            LogUtils.e(TAG, "Error creating input tensor: \(error.localizedDescription)")
            return nil
        }
    }
}
