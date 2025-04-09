import Foundation
import os.log

/**
 * Utility class for consistent logging throughout the SDK
 */
@objc public class LogUtils: NSObject {
    private static let SDK_TAG_PREFIX = "FaceSDK-"
    private static var isDebugEnabled = false
    
    /**
     * Enable or disable debug logging
     *
     * @param enabled True to enable debug logs, false to disable
     */
    @objc public static func setDebugEnabled(_ enabled: Bool) {
        isDebugEnabled = enabled
    }
    
    /**
     * Log a debug message
     *
     * @param tag Component tag
     * @param message Log message
     */
    @objc public static func d(_ tag: String, _ message: String) {
        if isDebugEnabled {
            if #available(iOS 14.0, *) {
                let logger = Logger(subsystem: "com.acleda.facelivenesssdk", category: "\(SDK_TAG_PREFIX)\(tag)")
                logger.debug("\(message)")
            } else {
                NSLog("[\(SDK_TAG_PREFIX)\(tag)] [DEBUG] \(message)")
            }
        }
    }
    
    /**
     * Log an info message
     *
     * @param tag Component tag
     * @param message Log message
     */
    @objc public static func i(_ tag: String, _ message: String) {
        if #available(iOS 14.0, *) {
            let logger = Logger(subsystem: "com.acleda.facelivenesssdk", category: "\(SDK_TAG_PREFIX)\(tag)")
            logger.info("\(message)")
        } else {
            NSLog("[\(SDK_TAG_PREFIX)\(tag)] [INFO] \(message)")
        }
    }
    
    /**
     * Log a warning message
     *
     * @param tag Component tag
     * @param message Log message
     */
    @objc public static func w(_ tag: String, _ message: String) {
        if #available(iOS 14.0, *) {
            let logger = Logger(subsystem: "com.acleda.facelivenesssdk", category: "\(SDK_TAG_PREFIX)\(tag)")
            logger.warning("\(message)")
        } else {
            NSLog("[\(SDK_TAG_PREFIX)\(tag)] [WARNING] \(message)")
        }
    }
    
    /**
     * Log an error message
     *
     * @param tag Component tag
     * @param message Log message
     */
    @objc public static func e(_ tag: String, _ message: String) {
        if #available(iOS 14.0, *) {
            let logger = Logger(subsystem: "com.acleda.facelivenesssdk", category: "\(SDK_TAG_PREFIX)\(tag)")
            logger.error("\(message)")
        } else {
            NSLog("[\(SDK_TAG_PREFIX)\(tag)] [ERROR] \(message)")
        }
    }
    
    /**
     * Log an error message with exception
     *
     * @param tag Component tag
     * @param message Log message
     * @param error Exception
     */
    @objc public static func e(_ tag: String, _ message: String, _ error: Error) {
        if #available(iOS 14.0, *) {
            let logger = Logger(subsystem: "com.acleda.facelivenesssdk", category: "\(SDK_TAG_PREFIX)\(tag)")
            logger.error("\(message): \(error.localizedDescription)")
        } else {
            NSLog("[\(SDK_TAG_PREFIX)\(tag)] [ERROR] \(message): \(error.localizedDescription)")
        }
    }
}
