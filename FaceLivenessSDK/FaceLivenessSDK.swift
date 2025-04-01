//
//  FaceLivenessSDK.swift
//  FaceLivenessSDK
//
//  Created by Sreang on 22/3/25.
//

import Foundation
import UIKit
import onnxruntime_objc

/**
 * Main SDK class for face liveness detection
 */
@objc public class FaceLivenessSDK: NSObject {
    private let tag = "FaceLivenessSDK"
    private let config: Config
    
    private lazy var livenessDetector: LivenessDetector = {
        return LivenessDetector()
    }()
    
    private lazy var occlusionDetector: FaceOcclusionDetector = {
        return FaceOcclusionDetector()
    }()
    
    /**
     * Configuration class for SDK initialization
     */
    @objc public class Config: NSObject {
        public let enableDebugLogging: Bool
        public let skipQualityCheck: Bool
        public let skipOcclusionCheck: Bool
        
        fileprivate init(enableDebugLogging: Bool, skipQualityCheck: Bool, skipOcclusionCheck: Bool) {
            self.enableDebugLogging = enableDebugLogging
            self.skipQualityCheck = skipQualityCheck
            self.skipOcclusionCheck = skipOcclusionCheck
            super.init()
        }
        
        /**
         * Builder class for SDK configuration
         */
        @objc public class Builder: NSObject {
            private var enableDebugLogging = false
            private var skipQualityCheck = false
            private var skipOcclusionCheck = false
            
            /**
             * Enable detailed debug logs
             */
            @objc public func setDebugLoggingEnabled(_ enabled: Bool) -> Builder {
                self.enableDebugLogging = enabled
                return self
            }
            
            /**
             * Skip image quality checks (not recommended for production)
             */
            @objc public func setSkipQualityCheck(_ skip: Bool) -> Builder {
                self.skipQualityCheck = skip
                return self
            }
            
            /**
             * Skip face occlusion checks (not recommended for production)
             */
            @objc public func setSkipOcclusionCheck(_ skip: Bool) -> Builder {
                self.skipOcclusionCheck = skip
                return self
            }
            
            /**
             * Build the configuration
             */
            @objc public func build() -> Config {
                return Config(
                    enableDebugLogging: enableDebugLogging,
                    skipQualityCheck: skipQualityCheck,
                    skipOcclusionCheck: skipOcclusionCheck
                )
            }
        }
    }
    
    // Private initializer
    private init(config: Config) {
        self.config = config
        super.init()
        
        // Configure logging based on config
        LogUtils.setDebugEnabled(config.enableDebugLogging)
        LogUtils.i(tag, "FaceLivenessSDK initialized with config: debugLogging=\(config.enableDebugLogging), " +
                 "skipQualityCheck=\(config.skipQualityCheck), skipOcclusionCheck=\(config.skipOcclusionCheck)")
    }
    
    /**
     * Create a new SDK instance with default configuration
     *
     * @return Configured SDK instance
     */
    @objc public class func create() -> FaceLivenessSDK {
        return create(config: Config.Builder().build())
    }
    
    /**
     * Create a new SDK instance with custom configuration
     *
     * @param config Custom configuration
     * @return Configured SDK instance
     */
    @objc public class func create(config: Config) -> FaceLivenessSDK {
        return FaceLivenessSDK(config: config)
    }
    
    /**
     * Full liveness detection process with occlusion check and quality check
     *
     * @param image Image to analyze
     * @param completion Callback with the liveness detection result
     */
    @objc public func detectLiveness(image: UIImage, completion: @escaping (FaceLivenessModel?, Error?) -> Void) {
        LogUtils.d(tag, "Starting face liveness detection process")
        
        // Validate input
        guard BitmapUtils.validateImage(image) else {
            completion(nil, InvalidImageException("Invalid input image"))
            return
        }
        
        // Perform detection in background
        DispatchQueue.global(qos: .userInitiated).async { [weak self] in
            guard let self = self else {
                completion(nil, FaceLivenessException("SDK instance was deallocated"))
                return
            }
            
            do {
                // Step 1: Occlusion check (if enabled)
                if !self.config.skipOcclusionCheck {
                    let occlusionResult = try self.occlusionDetector.detectFaceMask(image: image)
                    LogUtils.d(self.tag, "Face occlusion check result: \(occlusionResult.label) with confidence \(occlusionResult.confidence)")
                    
                    // If face is occluded (not normal), return as spoof
                    if occlusionResult.label != "normal" {
                        LogUtils.d(self.tag, "Face is occluded: \(occlusionResult.label), skipping further checks")
                        
                        let result = FaceLivenessModel(
                            prediction: "Spoof",
                            confidence: occlusionResult.confidence,
                            failureReason: "Face is occluded: \(occlusionResult.label)"
                        )
                        
                        DispatchQueue.main.async {
                            completion(result, nil)
                        }
                        return
                    }
                } else {
                    LogUtils.d(self.tag, "Face occlusion check skipped as per configuration")
                }
                
                // REMOVE CHECK QULITY
                
                // Step 3: Perform liveness detection
                LogUtils.d(self.tag, "Image quality acceptable, performing liveness detection")
                let detectionResult = try self.livenessDetector.runInference(image: image)
                
                let result = FaceLivenessModel(
                    prediction: detectionResult.label,
                    confidence: detectionResult.confidence
                )
                
                LogUtils.d(self.tag, "Detection complete: \(result)")
                
                DispatchQueue.main.async {
                    completion(result, nil)
                }
                
            } catch {
                LogUtils.e(self.tag, "Error in liveness detection pipeline: \(error.localizedDescription)")
                
                // Convert to SDK exception for consistent error handling
                let sdkError: Error
                if let _ = error as? FaceLivenessException {
                    sdkError = error
                } else {
                    sdkError = FaceLivenessException("SDK error: \(error.localizedDescription)")
                }
                
                DispatchQueue.main.async {
                    completion(nil, sdkError)
                }
            }
        }
    }
    
    /**
     * Check SDK version
     *
     * @return Version string
     */
    @objc public func getVersion() -> String {
        return "1.0.0"
    }
    
    /**
     * Cleanup resources when SDK is no longer needed
     */
    @objc public func close() {
        LogUtils.d(tag, "Closing FaceLivenessSDK resources")
        
        do {
            // Only access components that have been initialized
            if !config.skipOcclusionCheck {
                occlusionDetector.close()
            }
            
            livenessDetector.close()
        } catch {
            LogUtils.e(tag, "Error closing resources: \(error.localizedDescription)")
        }
    }
    
    deinit {
        LogUtils.d(tag, "FaceLivenessSDK instance deinitializing")
        close()
    }
}
