//  ModelUtils.swift
import Foundation
import onnxruntime_objc

/**
 * Utility functions for ML model handling
 */
@objc public class ModelUtils: NSObject {
    private static let TAG = "ModelUtils"
    
    /**
     * Load model from the bundle resources
     *
     * @param modelName Name of the model file in bundle
     * @return URL pointing to the model file
     * @throws ModelLoadingException if model loading fails
     */
    @objc public static func loadModelFromBundle(_ modelName: String) throws -> URL {
        guard let modelURL = Bundle(for: ModelUtils.self).url(forResource: modelName, withExtension: "onnx") else {
            LogUtils.e(TAG, "Could not find model \(modelName).onnx in bundle")
            throw ModelLoadingException("Failed to find model \(modelName).onnx in bundle")
        }
        
        LogUtils.d(TAG, "Model \(modelName) loaded successfully from: \(modelURL.path)")
        return modelURL
    }
    
    /**
     * Creates an ONNX Runtime session for the given model
     *
     * @param modelName Name of the ONNX model file (without extension)
     * @return OrtSession initialized with the model
     * @throws ModelLoadingException if session creation fails
     */
    @objc public static func createONNXSession(_ modelName: String) throws -> ORTSession {
        do {
            let modelURL = try loadModelFromBundle(modelName)
            
            // Create session options
            let sessionOptions = try ORTSessionOptions()
            
            // Enable optimization if needed
            try sessionOptions.setIntraOpNumThreads(1)
            try sessionOptions.setGraphOptimizationLevel(ORTGraphOptimizationLevel.all)
            
            // Create session
            let ortEnv = try ORTEnv(loggingLevel: ORTLoggingLevel.warning)
            let session = try ORTSession(env: ortEnv, modelPath: modelURL.path, sessionOptions: sessionOptions)
            
            LogUtils.d(TAG, "ONNX session created successfully for model: \(modelName)")
            return session
        } catch {
            LogUtils.e(TAG, "Error creating ONNX session for model \(modelName): \(error.localizedDescription)", error)
            throw ModelLoadingException("Failed to create ONNX session for model \(modelName): \(error.localizedDescription)", error)
        }
    }
    
    /**
     * Copies the ONNX model files from the bundle to a writable location
     * This is useful for first launch or if models need to be updated
     *
     * @return Bool indicating success
     */
    @objc public static func copyModelsFromBundleIfNeeded() -> Bool {
        let fileManager = FileManager.default
        let modelNames = ["FaceOcclusion", "Liveliness"]
        var success = true
        
        for modelName in modelNames {
            guard let bundleURL = Bundle(for: ModelUtils.self).url(forResource: modelName, withExtension: "onnx") else {
                LogUtils.e(TAG, "Model \(modelName).onnx not found in bundle")
                success = false
                continue
            }
            
            do {
                // Get documents directory for app
                let documentsURL = try fileManager.url(for: .documentDirectory,
                                                      in: .userDomainMask,
                                                      appropriateFor: nil,
                                                      create: true)
                let destURL = documentsURL.appendingPathComponent("\(modelName).onnx")
                
                // Only copy if the file doesn't exist or is outdated
                if !fileManager.fileExists(atPath: destURL.path) {
                    try fileManager.copyItem(at: bundleURL, to: destURL)
                    LogUtils.d(TAG, "Copied \(modelName).onnx to documents directory")
                }
            } catch {
                LogUtils.e(TAG, "Failed to copy \(modelName).onnx: \(error.localizedDescription)", error)
                success = false
            }
        }
        
        return success
    }
    
    /**
     * Get the input node name for an ONNX model
     *
     * @param session The ONNX session
     * @return The name of the input node
     */
    @objc public static func getInputName(_ session: ORTSession) throws -> String {
        do {
            let inputNames = try session.inputNames()
            guard let firstInput = inputNames.first else {
                throw ModelLoadingException("No input nodes found in model")
            }
            return firstInput
        } catch {
            LogUtils.e(TAG, "Error getting input name: \(error.localizedDescription)", error)
            throw ModelLoadingException("Failed to get input name: \(error.localizedDescription)", error)
        }
    }
    
    /**
     * Get the output node name for an ONNX model
     *
     * @param session The ONNX session
     * @return The name of the output node
     */
    @objc public static func getOutputName(_ session: ORTSession) throws -> String {
        do {
            let outputNames = try session.outputNames()
            guard let firstOutput = outputNames.first else {
                throw ModelLoadingException("No output nodes found in model")
            }
            return firstOutput
        } catch {
            LogUtils.e(TAG, "Error getting output name: \(error.localizedDescription)", error)
            throw ModelLoadingException("Failed to get output name: \(error.localizedDescription)", error)
        }
    }
}
