import Foundation
import UIKit
import onnxruntime_objc

public class FaceOcclusionDetector: NSObject {
    // MARK: - Constants
    private enum Constants {
        static let tag = "FaceOcclusionDetector"
        static let modelName = "FaceOcclusion" // Ensure this matches the actual file (e.g., "FaceOcclusion.onnx")
        static let imageSize = 224
        static let handOverFaceIndex = 0
        static let normalIndex = 1
        static let withMaskIndex = 2
        static let normalConfidenceThreshold: Float = 0.7
        static let expectedClassCount = 3
    }
    
    private let classNames: [Int: String] = [
        0: "hand_over_face",
        1: "normal",
        2: "with_mask"
    ]
    
    // MARK: - Properties
    private var ortSession: ORTSession?
    private var ortEnv: ORTEnv?
    private var isModelLoaded = false
    
    // MARK: - Initialization
    public override init() {
        super.init()
        loadModel()
    }
    
    // MARK: - Model Loading
    private func loadModel() {
        do {
            ortEnv = try ORTEnv(loggingLevel: .warning)
            guard let modelURL = Bundle.main.url(forResource: Constants.modelName, withExtension: "onnx") else {
                LogUtils.e(Constants.tag, "Failed to get model URL")
                isModelLoaded = false
                return
            }
            
            let sessionOptions = try ORTSessionOptions()
            try sessionOptions.setIntraOpNumThreads(1)
            try sessionOptions.setGraphOptimizationLevel(.all)
            
            ortSession = try ORTSession(
                env: ortEnv!,
                modelPath: modelURL.path,
                sessionOptions: sessionOptions
            )
            
            isModelLoaded = true
            LogUtils.d(Constants.tag, "Model loaded successfully from: \(modelURL.path)")
        } catch {
            isModelLoaded = false
            LogUtils.e(Constants.tag, "Error loading model: \(error.localizedDescription)", error)
        }
    }
    
    // MARK: - Public Interface
    public func detectFaceMask(image: UIImage) throws -> DetectionResult {
        LogUtils.d(Constants.tag, "Starting face occlusion detection")
        
        // Validate input image
        guard let cgImage = image.cgImage else {
            LogUtils.e(Constants.tag, "Invalid input image")
            throw FaceOcclusionError.invalidImage
        }
        
        // Check model state
        guard isModelLoaded, let session = ortSession else {
            LogUtils.w(Constants.tag, "Model not loaded, assuming normal face with low confidence")
            return DetectionResult(label: "normal", confidence: 0.7)
        }
        
        do {
            // Prepare input tensor
            let inputTensor = try prepareInputTensor(from: image)
            
            // Run inference
            let outputs = try runInference(session: session, inputTensor: inputTensor)
            
            // Process output
            return try processOutput(outputs: outputs)
        } catch {
            LogUtils.e(Constants.tag, "Error during inference: \(error.localizedDescription)", error)
            throw error
        }
    }
    
    public func reloadModel() -> Bool {
        guard !isModelLoaded else { return true }
        loadModel()
        return isModelLoaded
    }
    
    public func close() {
        ortSession = nil
        ortEnv = nil
        LogUtils.d(Constants.tag, "FaceOcclusionDetector resources released")
    }
    
    deinit {
        close()
    }
    
    // MARK: - Private Methods
    private func prepareInputTensor(from image: UIImage) throws -> ORTValue {
        // Resize to 224x224
        guard let resizedImage = resizeImage(image, toSize: CGSize(width: Constants.imageSize, height: Constants.imageSize)),
              let cgImage = resizedImage.cgImage else {
            throw FaceOcclusionError.imageNormalizationFailed
        }
        
        // Convert to RGB float buffer normalized to [0, 1]
        let width = Constants.imageSize
        let height = Constants.imageSize
        let pixelCount = width * height
        let bytesPerPixel = 4
        let data = UnsafeMutablePointer<UInt8>.allocate(capacity: pixelCount * bytesPerPixel)
        defer { data.deallocate() }
        
        let context = CGContext(
            data: data,
            width: width,
            height: height,
            bitsPerComponent: 8,
            bytesPerRow: width * bytesPerPixel,
            space: CGColorSpaceCreateDeviceRGB(),
            bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue
        )
        context?.draw(cgImage, in: CGRect(x: 0, y: 0, width: width, height: height))
        
        var floatArray = [Float](repeating: 0, count: 3 * pixelCount)
        for c in 0..<3 {
            for i in 0..<pixelCount {
                let value: Float = Float(data[i * bytesPerPixel + c]) / 255.0
                floatArray[c * pixelCount + i] = value
            }
        }
        
        let dataBuffer = Data(bytes: floatArray, count: floatArray.count * MemoryLayout<Float>.stride)
        let inputShape: [NSNumber] = [1, 3, NSNumber(value: Constants.imageSize), NSNumber(value: Constants.imageSize)]
        
        return try ORTValue(
            tensorData: NSMutableData(data: dataBuffer),
            elementType: .float,
            shape: inputShape
        )
    }
    
    private func resizeImage(_ image: UIImage, toSize size: CGSize) -> UIImage? {
        UIGraphicsBeginImageContextWithOptions(size, false, 1.0)
        image.draw(in: CGRect(origin: .zero, size: size))
        let resizedImage = UIGraphicsGetImageFromCurrentImageContext()
        UIGraphicsEndImageContext()
        return resizedImage
    }
    
    private func runInference(session: ORTSession, inputTensor: ORTValue) throws -> [String: ORTValue] {
        let inputNames = try session.inputNames()
        let outputNames = try session.outputNames()
        
        guard let inputName = inputNames.first, let outputName = outputNames.first else {
            throw FaceOcclusionError.modelIOFailure
        }
        
        let runOptions = try ORTRunOptions()
        return try session.run(
            withInputs: [inputName: inputTensor],
            outputNames: [outputName],
            runOptions: runOptions
        )
    }
    
    private func processOutput(outputs: [String: ORTValue]) throws -> DetectionResult {
        guard let outputTensor = outputs.values.first else {
            throw FaceOcclusionError.noOutputTensor
        }
        
        let floatArray = try extractProbabilities(from: outputTensor)
        logClassProbabilities(floatArray)
        
        let (maxIndex, maxProb) = findMaxProbability(floatArray)
        
        if maxIndex == Constants.normalIndex && maxProb < Constants.normalConfidenceThreshold {
            return handleLowConfidenceNormalCase(floatArray)
        }
        
        guard let className = classNames[maxIndex] else {
            throw FaceOcclusionError.unknownClass
        }
        
        return DetectionResult(label: className, confidence: maxProb)
    }
    
    private func extractProbabilities(from tensor: ORTValue) throws -> [Float] {
        guard let tensorData = try tensor.tensorData() as? NSData else {
            throw FaceOcclusionError.tensorDataExtractionFailed
        }
        
        let expectedByteCount = Constants.expectedClassCount * MemoryLayout<Float>.size
        guard tensorData.length == expectedByteCount else {
            throw FaceOcclusionError.invalidTensorSize(
                actual: tensorData.length,
                expected: expectedByteCount
            )
        }
        
        let floatBuffer = tensorData.bytes.assumingMemoryBound(to: Float.self)
        return Array(UnsafeBufferPointer(start: floatBuffer, count: Constants.expectedClassCount))
    }
    
    private func logClassProbabilities(_ probabilities: [Float]) {
        for (index, prob) in probabilities.enumerated() where index < classNames.count {
            let className = classNames[index] ?? "Unknown"
            LogUtils.d(Constants.tag, "Class \(className): \(prob)")
        }
    }
    
    private func findMaxProbability(_ probabilities: [Float]) -> (index: Int, probability: Float) {
        var maxIndex = 0
        var maxProb: Float = 0.0
        
        for (index, prob) in probabilities.enumerated() where index < classNames.count {
            if prob > maxProb {
                maxProb = prob
                maxIndex = index
            }
        }
        
        return (maxIndex, maxProb)
    }
    
    private func handleLowConfidenceNormalCase(_ probabilities: [Float]) -> DetectionResult {
        let maskProb = probabilities[Constants.withMaskIndex]
        let handOverFaceProb = probabilities[Constants.handOverFaceIndex]
        
        if maskProb > handOverFaceProb {
            LogUtils.d(Constants.tag, "Reassigned to with_mask with probability: \(maskProb)")
            return DetectionResult(label: "with_mask", confidence: maskProb)
        } else {
            LogUtils.d(Constants.tag, "Reassigned to hand_over_face with probability: \(handOverFaceProb)")
            return DetectionResult(label: "hand_over_face", confidence: handOverFaceProb)
        }
    }
}

// MARK: - Error Handling
extension FaceOcclusionDetector {
    public enum FaceOcclusionError: Error {
        case invalidImage
        case modelNotLoaded
        case imageNormalizationFailed
        case modelIOFailure
        case noOutputTensor
        case tensorDataExtractionFailed
        case invalidTensorSize(actual: Int, expected: Int)
        case unknownClass
        
        var localizedDescription: String {
            switch self {
            case .invalidImage: return "Invalid input image"
            case .modelNotLoaded: return "Model not loaded"
            case .imageNormalizationFailed: return "Failed to normalize image"
            case .modelIOFailure: return "Failed to get model input/output names"
            case .noOutputTensor: return "No output tensor from model"
            case .tensorDataExtractionFailed: return "Failed to extract tensor data"
            case .invalidTensorSize(let actual, let expected):
                return "Invalid tensor size (actual: \(actual), expected: \(expected))"
            case .unknownClass: return "Unknown class detected"
            }
        }
    }
}


