import Foundation
import UIKit
import onnxruntime_objc

/**
 * Detects face occlusions such as masks or hands covering the face
 */
@objc public class FaceOcclusionDetector: NSObject {
    // Constants
    private let TAG = "FaceOcclusionDetector"
    private let MODEL_NAME = "FaceOcclusion"
    private let IMAGE_SIZE = 224
    private let HAND_OVER_FACE_INDEX = 0
    private let NORMAL_INDEX = 1
    private let WITH_MASK_INDEX = 2
    private let NORMAL_CONFIDENCE_THRESHOLD: Float = 0.7
    
    // Class mapping
    private let classNames: [Int: String] = [
        0: "hand_over_face",
        1: "normal",
        2: "with_mask"
    ]
    
    private var ortSession: ORTSession?
    private var ortEnv: ORTEnv?
    private var isModelLoaded = false
    
    /**
     * Initialize the detector
     */
    public override init() {
        super.init()
        loadModel()
    }
    
    /**
     * Load the ONNX model
     */
    private func loadModel() {
        do {
            // Create environment
            ortEnv = try ORTEnv(loggingLevel: ORTLoggingLevel.warning)
            
            guard let modelURL = try? ModelUtils.loadModelFromBundle(MODEL_NAME) else {
                LogUtils.e(TAG, "Failed to get model URL")
                isModelLoaded = false
                return
            }
            
            // Create session options
            let sessionOptions = try ORTSessionOptions()
            try sessionOptions.setIntraOpNumThreads(1)
            try sessionOptions.setGraphOptimizationLevel(ORTGraphOptimizationLevel.all)
            
            // Create session
            ortSession = try ORTSession(env: ortEnv!, modelPath: modelURL.path, sessionOptions: sessionOptions)
            isModelLoaded = true
            LogUtils.d(TAG, "Model loaded successfully from: \(modelURL.path)")
        } catch {
            isModelLoaded = false
            LogUtils.e(TAG, "Error loading model: \(error.localizedDescription)", error)
        }
    }
    
    /**
     * Detect if face is occluded by mask or hand
     *
     * @param image Image to analyze
     * @return DetectionResult containing class name and confidence
     * @throws OcclusionDetectionException if detection fails
     */
    @objc public func detectFaceMask(image: UIImage) throws -> DetectionResult {
        LogUtils.d(TAG, "Starting face occlusion detection")
        
        // Validate input
        guard BitmapUtils.validateImage(image) else {
            LogUtils.e(TAG, "Invalid input image")
            throw InvalidImageException("Invalid input image")
        }
        
        // If model failed to load, return normal with low confidence
        // This allows the pipeline to continue instead of failing
        guard isModelLoaded, let session = ortSession, let env = ortEnv else {
            LogUtils.w(TAG, "Model not loaded, assuming normal face with low confidence")
            return DetectionResult(label: "normal", confidence: 0.7)
        }
        
        do {
            // Prepare input tensor
            let inputNames = try session.inputNames()
            let outputNames = try session.outputNames()
            
            guard let inputName = inputNames.first, let outputName = outputNames.first else {
                throw OcclusionDetectionException("Failed to get input/output names")
            }
            
            // Normalize image for the model
            // Using ImageNet normalization values
            guard let normalizedImageData = BitmapUtils.normalizeImage(
                image,
                width: IMAGE_SIZE,
                height: IMAGE_SIZE,
                means: [0.485, 0.456, 0.406],
                stds: [0.229, 0.224, 0.225]
            ) else {
                throw OcclusionDetectionException("Failed to normalize image")
            }
            
            // Convert to Data
            let data = Data(bytes: normalizedImageData, count: normalizedImageData.count * MemoryLayout<Float>.stride)
            let nsData = data as NSData
            
            // Create shape array
            let inputShape: [NSNumber] = [1, 3, NSNumber(value: IMAGE_SIZE), NSNumber(value: IMAGE_SIZE)]
            
            // Create input tensor - adjust this based on your API
            let inputTensor = try ORTValue(tensorData: nsData as! NSMutableData,
                                          elementType: ORTTensorElementDataType.float,
                                          shape: inputShape)
            
            // Create run options
            let runOptions = try ORTRunOptions()
            
            // Run inference
            let inputs = [inputName: inputTensor]
            let outputs = try session.run(
                withInputs: inputs,
                outputNames: [outputName],
                runOptions: runOptions
            )
            
            guard let outputTensor = outputs[outputName] else {
                throw OcclusionDetectionException("No output tensor")
            }
            
            // Extract results - adjust based on your API
            let floatArray = try extractFloatArray(from: outputTensor)
            
            // Log all probabilities for debugging
            for (index, prob) in floatArray.enumerated() {
                if index < classNames.count {
                    LogUtils.d(TAG, "Class \(classNames[index] ?? "Unknown"): \(prob)")
                }
            }
            
            // Find max probability class
            var maxIndex = 0
            var maxProb: Float = 0.0
            
            for i in 0..<min(floatArray.count, classNames.count) {
                if floatArray[i] > maxProb {
                    maxProb = floatArray[i]
                    maxIndex = i
                }
            }
            
            // Apply the custom condition:
            // If predicted class is "normal" but confidence < threshold,
            // choose either "with_mask" or "hand_over_face" based on highest probability
            if maxIndex == NORMAL_INDEX && maxProb < NORMAL_CONFIDENCE_THRESHOLD {
                LogUtils.d(TAG, "Normal class detected with low confidence: \(maxProb), reassigning...")
                
                // Get the probabilities of the other two classes
                let maskProb = floatArray[WITH_MASK_INDEX]
                let handOverFaceProb = floatArray[HAND_OVER_FACE_INDEX]
                
                // Choose the class with higher probability between mask and hand over face
                if maskProb > handOverFaceProb {
                    LogUtils.d(TAG, "Reassigned to with_mask with probability: \(maskProb)")
                    return DetectionResult(label: "with_mask", confidence: maskProb)
                } else {
                    LogUtils.d(TAG, "Reassigned to hand_over_face with probability: \(handOverFaceProb)")
                    return DetectionResult(label: "hand_over_face", confidence: handOverFaceProb)
                }
            }
            
            // Standard case - return the highest probability class
            let className = classNames[maxIndex] ?? "Unknown"
            return DetectionResult(label: className, confidence: maxProb)
            
        } catch {
            LogUtils.e(TAG, "Error during inference: \(error.localizedDescription)", error)
            throw OcclusionDetectionException("Error during occlusion detection: \(error.localizedDescription)", error)
        }
    }
    
    // Helper method to extract float values from tensor - adjust based on your API
    private func extractFloatArray(from tensor: ORTValue) throws -> [Float] {
        // Get tensor data as NSArray
        guard let data = try tensor.tensorData() as? [NSNumber] else {
            throw OcclusionDetectionException("Failed to get tensor data as array")
        }

        // Convert NSNumber array to Float array
        return data.map { $0.floatValue }
    }
    
    /**
     * Try to reload model if it failed to load initially
     *
     * @return true if model loaded successfully
     */
    @objc public func reloadModel() -> Bool {
        if isModelLoaded { return true }
        
        loadModel()
        return isModelLoaded
    }
    
    /**
     * Close and release resources
     */
    @objc public func close() {
        ortSession = nil
        ortEnv = nil
        LogUtils.d(TAG, "FaceOcclusionDetector resources released")
    }
    
    deinit {
        close()
    }
}
