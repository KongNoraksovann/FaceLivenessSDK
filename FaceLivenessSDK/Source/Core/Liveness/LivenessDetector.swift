import Foundation
import UIKit
import onnxruntime_objc

/**
 * Handles face liveness detection using ONNX runtime
 */
@objc public class LivenessDetector: NSObject {
    private let TAG = "LivenessDetector"
    
    // Constants
    private let MODEL_PATH = "Liveliness"
    private let INPUT_SIZE = 224
    private let LIVE_THRESHOLD: Float = 0.5
    
    // ImageNet normalization values
    private let mean: [Float] = [0.485, 0.456, 0.406]
    private let std: [Float] = [0.229, 0.224, 0.225]
    
    private var session: ORTSession?
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
            
            guard let modelURL = try? ModelUtils.loadModelFromBundle(MODEL_PATH) else {
                LogUtils.e(TAG, "Failed to get model URL")
                isModelLoaded = false
                return
            }
            
            // Create session options
            let sessionOptions = try ORTSessionOptions()
            try sessionOptions.setIntraOpNumThreads(1)
            try sessionOptions.setGraphOptimizationLevel(ORTGraphOptimizationLevel.all)
            
            // Create session
            session = try ORTSession(env: ortEnv!, modelPath: modelURL.path, sessionOptions: sessionOptions)
            isModelLoaded = true
            LogUtils.d(TAG, "Liveness model loaded successfully")
        } catch {
            isModelLoaded = false
            LogUtils.e(TAG, "Error loading liveness model: \(error.localizedDescription)", error)
        }
    }
    
    /**
     * Run face liveness detection on the provided image
     *
     * @param image The image to analyze (should be a cropped face)
     * @return DetectionResult containing the result label ("Live" or "Spoof") and confidence
     * @throws LivenessException if detection fails
     */
    @objc public func runInference(image: UIImage) throws -> DetectionResult {
        LogUtils.d(TAG, "Starting liveness inference on face image: \(Int(image.size.width))x\(Int(image.size.height))")
        
        // Validate input
        guard BitmapUtils.validateImage(image) else {
            LogUtils.e(TAG, "Invalid input image")
            throw InvalidImageException("Invalid input image")
        }
        
        // If model failed to load, return a default result
        guard isModelLoaded, let session = session, let env = ortEnv else {
            LogUtils.e(TAG, "Model not loaded")
            throw LivenessException("Liveness model not loaded")
        }
        
        do {
            // Prepare input tensor
            let inputNames = try session.inputNames()
            let outputNames = try session.outputNames()
            
            guard let inputName = inputNames.first, let outputName = outputNames.first else {
                throw LivenessException("Failed to get input/output names")
            }
            
            // Normalize image for the model
            guard let normalizedImageData = BitmapUtils.normalizeImage(
                image,
                width: INPUT_SIZE,
                height: INPUT_SIZE,
                means: mean,
                stds: std
            ) else {
                throw LivenessException("Failed to normalize image")
            }
            
            // Convert to Data
            let data = Data(bytes: normalizedImageData, count: normalizedImageData.count * MemoryLayout<Float>.stride)
            let nsData = data as NSData
            
            // Create shape array
            let inputShape: [NSNumber] = [1, 3, NSNumber(value: INPUT_SIZE), NSNumber(value: INPUT_SIZE)]
            
            // Create input tensor - using the constructor that works with your API
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
                throw LivenessException("No output tensor produced")
            }
            
            // Extract logit value using the method that works with your API
            let floatArray = try extractFloatArray(from: outputTensor)
            
            guard !floatArray.isEmpty else {
                throw LivenessException("Empty output array")
            }
            
            let logit = floatArray[0]
            LogUtils.d(TAG, "Raw model output (logit): \(logit)")
            
            // Apply sigmoid to get confidence score
            let conf = 1.0 / (1.0 + exp(-logit))
            LogUtils.d(TAG, "Confidence after sigmoid: \(conf)")
            
            // Apply threshold for classification
            let label = conf > LIVE_THRESHOLD ? "Live" : "Spoof"
            
            // Adjust confidence display (showing confidence in the prediction)
            let displayConf = label == "Live" ? conf : 1.0 - conf
            LogUtils.d(TAG, "Final prediction: \(label) with display confidence: \(displayConf)")
            
            return DetectionResult(label: label, confidence: displayConf)
            
        } catch {
            LogUtils.e(TAG, "Error during inference: \(error.localizedDescription)", error)
            throw LivenessException("Error during liveness detection: \(error.localizedDescription)", error)
        }
    }
    
    // Helper method to extract float values from tensor
    private func extractFloatArray(from tensor: ORTValue) throws -> [Float] {
        // Get tensor data as NSArray
        guard let data = try tensor.tensorData() as? [NSNumber] else {
            throw LivenessException("Failed to get tensor data as array")
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
        session = nil
        ortEnv = nil
        LogUtils.d(TAG, "LivenessDetector resources released")
    }
    
    deinit {
        close()
    }
}
