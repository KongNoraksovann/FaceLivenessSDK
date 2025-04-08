//  LivenessDetector.swift
import Foundation
import UIKit
import onnxruntime_objc
/**
 * Handles face liveness detection using ONNX runtime
 */
@objc public class LivenessDetector: NSObject {
    private let TAG = "LivenessDetector"
    
    // Constants
    private let MODEL_NAME = "Liveliness"
    private let INPUT_SIZE = 224
    private let LIVE_THRESHOLD: Float = 0.5
    
    // ImageNet normalization values
    private let mean: [Float] = [0.485, 0.456, 0.406]
    private let std: [Float] = [0.229, 0.224, 0.225]
    
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
     * Run face liveness detection on the provided image
     *
     * @param image The image to analyze (should be a cropped face)
     * @return Dictionary containing "label" (String) and "confidence" (Float) keys, or nil if detection fails
     */
    @objc public func runInference(image: UIImage) -> [String: Any]? {
        LogUtils.d(TAG, "Starting liveness inference on face image: \(Int(image.size.width))x\(Int(image.size.height))")
        
        // Validate input
        guard BitmapUtils.validateImage(image) else {
            LogUtils.e(TAG, "Invalid input image")
            return nil
        }
        
        // If model failed to load
        guard isModelLoaded, let session = ortSession else {
            LogUtils.e(TAG, "Model not loaded")
            return nil
        }
        
        do {
            // Prepare input tensor
            guard let inputNames = try? session.inputNames(),
                  let inputName = inputNames.first else {
                LogUtils.e(TAG, "No input name found")
                return nil
            }
            
            // Normalize image for the model
            guard let normalizedImageData = BitmapUtils.normalizeImage(
                image,
                width: INPUT_SIZE,
                height: INPUT_SIZE,
                means: mean,
                stds: std
            ) else {
                LogUtils.e(TAG, "Failed to normalize image")
                return nil
            }
            
            // Create properly formatted mutable data for tensor input
            let dataLength = normalizedImageData.count * MemoryLayout<Float>.stride
            let nsData = NSMutableData(bytes: normalizedImageData, length: dataLength)
            
            // Create shape array
            let inputShape: [NSNumber] = [1, 3, NSNumber(value: INPUT_SIZE), NSNumber(value: INPUT_SIZE)]
            
            // Create input tensor
            guard let inputTensor = try? ORTValue(tensorData: nsData,
                                                elementType: ORTTensorElementDataType.float,
                                                shape: inputShape) else {
                LogUtils.e(TAG, "Failed to create input tensor")
                return nil
            }
            
            // Run inference
            LogUtils.d(TAG, "Running model inference")
            let outputNames: Set<String> = ["output"]
            let inputs = [inputName: inputTensor]
            
            let outputs: [String: ORTValue]
            do {
                outputs = try session.run(withInputs: inputs,
                                        outputNames: outputNames,
                                        runOptions: nil)
            } catch {
                LogUtils.e(TAG, "Inference failed: \(error.localizedDescription)")
                return nil
            }
            
            // Process output
            guard let outputTensor = outputs["output"],
                  let outputData = try? outputTensor.tensorData() else {
                LogUtils.e(TAG, "Failed to get output tensor or data")
                return nil
            }
            
            let outputBuffer = outputData.bytes.bindMemory(to: Float.self,
                                                         capacity: outputData.length / MemoryLayout<Float>.size)
            let outputSize = outputData.length / MemoryLayout<Float>.size
            
            guard outputSize > 0 else {
                LogUtils.e(TAG, "Empty output from model")
                return nil
            }
            
            let logit = outputBuffer[0]
            LogUtils.d(TAG, "Raw model output (logit): \(logit)")
            
            let conf = Float(1.0 / (1.0 + exp(-Double(logit))))
            LogUtils.d(TAG, "Confidence after sigmoid: \(conf)")
            
            let label = conf > LIVE_THRESHOLD ? "Live" : "Spoof"
            let displayConf = label == "Live" ? conf : 1.0 - conf
            LogUtils.d(TAG, "Final prediction: \(label) with display confidence: \(displayConf)")
            
            return ["label": label, "confidence": displayConf]
            
        } catch {
            LogUtils.e(TAG, "Error during inference: \(error.localizedDescription)")
            return nil
        }
    }
    
    // Helper method to extract float values from tensor
    private func extractFloatArray(from tensor: ORTValue) throws -> [Float] {
        do {
            // First, try using the tensor data as an array of NSNumber
            if let data = try tensor.tensorData() as? [NSNumber] {
                return data.map { $0.floatValue }
            }
            
            // Alternative method: get raw data and convert to float array
            if let tensorData = try tensor.tensorData() as? NSData {
                let count = tensorData.length / MemoryLayout<Float>.stride
                var floatArray = [Float](repeating: 0, count: count)
                
                tensorData.getBytes(&floatArray, length: tensorData.length)
                return floatArray
            }
            
            throw LivenessException("Failed to extract data from tensor")
        } catch {
            LogUtils.e(TAG, "Error extracting tensor data: \(error.localizedDescription)")
            throw LivenessException("Failed to extract tensor data: \(error.localizedDescription)")
        }
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
        LogUtils.d(TAG, "LivenessDetector resources released")
    }
    
    deinit {
        close()
    }
}
